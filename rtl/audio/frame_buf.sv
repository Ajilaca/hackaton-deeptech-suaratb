// frame_buf: penyangga sampel pasca pre-emphasis, pembangkit frame 400 sampel tiap hop 160, dan
//            penghitung energi hop untuk detektor.
// Fungsi    : cincin RAM 512 x W. Setelah sampel ke-400 (lalu tiap 160 sampel) memutar ulang 400 sampel
//             terakhir sebagai stream 1 sampel/siklus (frame_first..frame_last). Bersamaan menghasilkan
//             e_out = sum(s16^2) atas 160 sampel terbaru (jendela [160f+240, 160f+400)), e_valid 1 siklus
//             SEBELUM frame_valid. Zero-pad ke 512 dan window dikerjakan modul window (Fase 3).
// Parameter : W (lebar sampel lebar = 16+FRAC), FRAME_LEN, HOP, NRING(512), E_W (48)
// Antarmuka : in_valid/in_wide/in_s16 (stream valid saja). Sampel masukan HARUS berjarak >= FRAME_LEN+4 siklus.
//             keluaran frame_valid/frame_data/frame_first/frame_last/frame_idx; e_valid/e_out.
// Latensi   : e_valid 2 siklus, frame_valid (frame_first) 3 siklus setelah sampel pemicu (kuadrat didaftarkan lebih dulu).
// Golden    : golden/blocks/frontend.py frame_signal(preemph(x, frac)), cough_det.py hop_energy(preemph(x))
module frame_buf #(
  parameter int W         = 16 + suaratb_pkg::PREEMPH_FRAC,
  parameter int FRAME_LEN = suaratb_pkg::FRAME_LEN,
  parameter int HOP       = suaratb_pkg::HOP,
  parameter int NRING     = 512,
  parameter int E_W       = 48
) (
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    in_valid,
  input  logic signed [W-1:0]     in_wide,
  input  logic signed [15:0]      in_s16,
  output logic                    frame_valid,
  output logic signed [W-1:0]     frame_data,
  output logic                    frame_first,
  output logic                    frame_last,
  output logic [31:0]             frame_idx,
  output logic                    e_valid,
  output logic [E_W-1:0]          e_out
);
  localparam int AW       = $clog2(NRING);
  localparam int PRE_SKIP = FRAME_LEN - HOP;       // sampel yang tak ikut jendela energi pertama (240)

  logic [AW-1:0]        wr_ptr;
  logic [AW-1:0]        rd_base;
  logic [AW-1:0]        raddr;
  logic [W-1:0]         rdata;
  logic                 rd_en;

  logic [8:0]           cnt_pre;
  logic                 acc_on;
  logic [8:0]           win_pos;
  logic [E_W-1:0]       acc;
  logic                 start;                    // pemicu frame (1 siklus)

  logic                 rd_active;
  logic [8:0]           rd_idx;
  logic                 v_d1, first_d1, last_d1;
  logic [31:0]          fidx;

  wire [31:0] sq = in_s16 * in_s16;               // bertanda x bertanda -> 32 bit tak negatif

  // RAM cincin
  ram_sdp #(.WIDTH(W), .DEPTH(NRING)) u_ram (
    .clk(clk), .we(in_valid), .waddr(wr_ptr), .wdata(in_wide),
    .re(rd_en), .raddr(raddr), .rdata(rdata)
  );

  // penulisan, energi hop, pemicu frame. Kuadrat sampel didaftarkan (sq_q) sebelum diakumulasi supaya
  // jalur pengali 16x16 -> penjumlah 48 bit terpisah menjadi dua tahap.
  logic        acc_en_q, acc_last_q;
  logic [31:0] sq_q;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      wr_ptr  <= '0;
      cnt_pre <= '0;
      acc_on  <= 1'b0;
      win_pos <= '0;
      acc     <= '0;
      e_out   <= '0;
      e_valid <= 1'b0;
      start   <= 1'b0;
      acc_en_q   <= 1'b0;
      acc_last_q <= 1'b0;
    end else begin
      e_valid    <= 1'b0;
      start      <= 1'b0;
      acc_en_q   <= 1'b0;
      acc_last_q <= 1'b0;
      if (in_valid) begin
        wr_ptr <= wr_ptr + 1'b1;
        if (!acc_on) begin
          if (cnt_pre == PRE_SKIP - 1) acc_on <= 1'b1;
          else                          cnt_pre <= cnt_pre + 1'b1;
        end else begin
          sq_q     <= sq;
          acc_en_q <= 1'b1;
          if (win_pos == HOP - 1) begin
            acc_last_q <= 1'b1;
            win_pos    <= '0;
            start      <= 1'b1;
          end else begin
            win_pos <= win_pos + 1'b1;
          end
        end
      end
      // tahap akumulasi (satu siklus setelah sampel)
      if (acc_en_q) begin
        if (acc_last_q) begin
          e_out   <= acc + {{(E_W-32){1'b0}}, sq_q};
          e_valid <= 1'b1;
          acc     <= '0;
        end else begin
          acc <= acc + {{(E_W-32){1'b0}}, sq_q};
        end
      end
    end
  end

  // pembacaan frame: mulai 1 siklus setelah start (wr_ptr sudah bertambah)
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rd_active <= 1'b0;
      rd_idx    <= '0;
      rd_base   <= '0;
      fidx      <= '0;
      v_d1      <= 1'b0;
      first_d1  <= 1'b0;
      last_d1   <= 1'b0;
    end else begin
      v_d1     <= rd_active;
      first_d1 <= rd_active && (rd_idx == 0);
      last_d1  <= rd_active && (rd_idx == FRAME_LEN - 1);
      if (start) begin
        rd_active <= 1'b1;
        rd_idx    <= '0;
        rd_base   <= wr_ptr - AW'(FRAME_LEN);
      end else if (rd_active) begin
        if (rd_idx == FRAME_LEN - 1) begin
          rd_active <= 1'b0;
          fidx      <= fidx + 1'b1;
        end else begin
          rd_idx <= rd_idx + 1'b1;
        end
      end
    end
  end

  assign rd_en = rd_active;
  assign raddr = rd_base + rd_idx[AW-1:0];

  assign frame_valid = v_d1;
  assign frame_data  = rdata;
  assign frame_first = first_d1;
  assign frame_last  = last_d1;
  assign frame_idx   = last_d1 ? fidx - 1'b1 : fidx;   // fidx naik setelah siklus terakhir dibaca

`ifndef SYNTHESIS
  always_ff @(posedge clk) begin
    if (rst_n && rd_active && in_valid)
      $error("frame_buf: sampel masukan tiba saat frame sedang dibaca (jarak < FRAME_LEN+4)");
  end
`endif
endmodule
