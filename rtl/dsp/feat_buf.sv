// feat_buf: penyangga cincin fitur int8 (RING_FRAMES x N_MEL) dan pembaca segmen 50 frame untuk CNN.
// Fungsi    : TULIS : tiap frame = N_MEL nilai (in_idx 0..39, in_last pada idx 39); frame_count bertambah per frame.
//             BACA  : seg_go (1 siklus) dengan seg_start (indeks frame absolut) -> N_FRAMES x N_MEL nilai dalam urutan
//                     CNN (mel-major: m = 0..39 luar, f = 0..49 dalam), out_addr = m*N_FRAMES + f (0..1999).
//             Segmen harus lengkap (seg_start + N_FRAMES <= frame_count) dan masih di dalam cincin
//             (frame_count - seg_start <= RING_FRAMES); frame di luar rentang tidak dikoreksi (asersi di simulasi).
// Parameter : N_MEL, N_FRAMES, RING_FRAMES (pangkat 2, >= N_FRAMES + 1)
// Antarmuka : in_valid/in_q/in_idx/in_last; seg_go/seg_start; keluaran out_valid/out_q/out_addr/out_last, seg_busy,
//             frame_count (jumlah frame tertulis)
// Latensi   : seg_go -> out_valid 3 siklus; 1 nilai/siklus (N_MEL*N_FRAMES siklus)
// Golden    : golden/blocks/frontend.py : segment(Lq, start) lalu to_cnn_input(...) (flatten (1, 40, 50))
module feat_buf #(
  parameter int N_MEL       = suaratb_pkg::N_MEL,
  parameter int N_FRAMES    = suaratb_pkg::N_FRAMES,
  parameter int RING_FRAMES = 64
) (
  input  logic                clk,
  input  logic                rst_n,
  input  logic                in_valid,
  input  logic signed [7:0]   in_q,
  input  logic [5:0]          in_idx,
  input  logic                in_last,
  input  logic                seg_go,
  input  logic [31:0]         seg_start,
  output logic                out_valid,
  output logic signed [7:0]   out_q,
  output logic [10:0]         out_addr,
  output logic                out_last,
  output logic                seg_busy,
  output logic [31:0]         frame_count
);
  localparam int DEPTH = RING_FRAMES * N_MEL;
  localparam int AW    = $clog2(DEPTH);
  localparam int FB    = $clog2(RING_FRAMES);

  // ---- tulis ---------------------------------------------------------------------------------------
  logic [AW-1:0] wr_addr;
  logic [FB-1:0] wr_frame;
  wire  [AW-1:0] wr_a = (wr_frame * N_MEL) + AW'(in_idx);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      wr_frame    <= '0;
      frame_count <= '0;
    end else if (in_valid && in_last) begin
      wr_frame    <= wr_frame + 1'b1;               // RING_FRAMES pangkat 2: membungkus sendiri
      frame_count <= frame_count + 32'd1;
    end
  end

  // ---- baca ----------------------------------------------------------------------------------------
  logic [5:0]    m_cnt;
  logic [5:0]    f_cnt;
  logic [FB-1:0] f_base;
  logic          rd_run;
  logic [10:0]   o_cnt;
  logic          rd_en;
  logic [AW-1:0] rd_addr;
  logic signed [7:0] rd_data;

  wire [FB-1:0] fp = f_base + FB'(f_cnt);
  assign rd_en   = rd_run;
  assign rd_addr = (fp * N_MEL) + AW'(m_cnt);
  assign seg_busy = rd_run;

  ram_sdp #(.WIDTH(8), .DEPTH(DEPTH)) u_ram (
    .clk(clk), .we(in_valid), .waddr(wr_a), .wdata(in_q),
    .re(rd_en), .raddr(rd_addr), .rdata(rd_data)
  );

  logic          v1, last1;
  logic [10:0]   a1;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rd_run    <= 1'b0;
      m_cnt     <= '0;
      f_cnt     <= '0;
      o_cnt     <= '0;
      f_base    <= '0;
      v1        <= 1'b0;
      out_valid <= 1'b0;
      out_last  <= 1'b0;
    end else begin
      v1        <= rd_run;
      last1     <= rd_run && (m_cnt == 6'(N_MEL - 1)) && (f_cnt == 6'(N_FRAMES - 1));
      a1        <= o_cnt;
      out_valid <= v1;
      out_last  <= v1 && last1;
      if (seg_go && !rd_run) begin
        rd_run <= 1'b1;
        m_cnt  <= '0;
        f_cnt  <= '0;
        o_cnt  <= '0;
        f_base <= seg_start[FB-1:0];
      end else if (rd_run) begin
        o_cnt <= o_cnt + 11'd1;
        if (f_cnt == 6'(N_FRAMES - 1)) begin
          f_cnt <= '0;
          if (m_cnt == 6'(N_MEL - 1)) rd_run <= 1'b0;
          else                        m_cnt <= m_cnt + 6'd1;
        end else begin
          f_cnt <= f_cnt + 6'd1;
        end
      end
    end
    if (v1) begin
      out_q    <= rd_data;
      out_addr <= a1;
    end
  end

`ifndef SYNTHESIS
  always_ff @(posedge clk) begin
    if (rst_n && seg_go) begin
      if (seg_start + N_FRAMES > frame_count)
        $error("feat_buf: segmen belum lengkap (start=%0d, frame_count=%0d)", seg_start, frame_count);
      if (frame_count - seg_start > RING_FRAMES)
        $error("feat_buf: segmen sudah tertimpa cincin (start=%0d, frame_count=%0d)", seg_start, frame_count);
    end
  end
`endif
endmodule
