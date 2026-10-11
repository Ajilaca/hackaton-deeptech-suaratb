// window: jendela Hamming 400 titik (Q1.15) + zero-pad ke 512 titik, keluaran W bit bertanda.
// Fungsi    : y[n] = sat_W( round(x[n] * w[n] / 2^(WIN_FRAC-G)) ), G = W - 16 - IN_FRAC, n = 0..399;
//             y[n] = 0 untuk n = 400..511 (dikeluarkan setelah sampel ke-400, 1 sampel/siklus).
//             ROM menyimpan 200 entri (jendela simetris): alamat = n < 200 ? n : 399 - n.
//             round(v/2^s) = (v + 2^(s-1)) >>> s (tambah setengah lalu geser aritmetik).
// Parameter : W, IN_FRAC (bit pecahan masukan = PREEMPH_FRAC), WIN_FRAC, FRAME_LEN, NFFT, INIT_FILE
// Antarmuka : in_valid/in_data/in_first (masukan 16+IN_FRAC bit, stream berurutan tanpa celah 400 sampel)
//             -> out_valid/out_data/out_first/out_last (512 sampel berurutan, out_last pada n = 511)
// Latensi   : 3 siklus (sampel pertama); nol-pad menyusul tanpa celah. Frame berikutnya tidak boleh tiba
//             sebelum 512 keluaran selesai (frame_buf: jarak frame = 160 sampel audio >> 512 siklus).
// Golden    : golden/blocks/window.py : apply_window(frames, W, in_frac)
module window #(
  parameter int    W         = suaratb_pkg::FFT_W,
  parameter int    IN_FRAC   = suaratb_pkg::PREEMPH_FRAC,
  parameter int    WIN_FRAC  = suaratb_pkg::WIN_FRAC,
  parameter int    FRAME_LEN = suaratb_pkg::FRAME_LEN,
  parameter int    NFFT      = suaratb_pkg::NFFT,
  parameter INIT_FILE = "rtl/mem/window_hamming.mem"
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         in_valid,
  input  logic signed [16+IN_FRAC-1:0] in_data,
  input  logic                         in_first,
  output logic                         out_valid,
  output logic signed [W-1:0]          out_data,
  output logic                         out_first,
  output logic                         out_last
);
  localparam int IN_W  = 16 + IN_FRAC;
  localparam int G     = W - 16 - IN_FRAC;          // bit pecahan tambahan di atas masukan
  localparam int SH    = WIN_FRAC - G;              // geser pembulatan (>= 1)
  localparam int HALF  = FRAME_LEN / 2;             // 200
  localparam int NPAD  = NFFT - FRAME_LEN;          // 112
  localparam int PW    = IN_W + 17;                 // lebar hasil kali (bobot 17 bit bertanda)

`ifndef SYNTHESIS
  initial begin
    if (G < 0 || G > WIN_FRAC - 1) $error("window: G=%0d di luar [0, WIN_FRAC-1]", G);
  end
`endif

  // ---- tahap 0: penghitung indeks sampel dan alamat ROM --------------------------------------------
  logic [8:0] n_cnt;
  wire  [8:0] n_cur = in_first ? 9'd0 : n_cnt;
  wire        s0_last = (n_cur == 9'(FRAME_LEN - 1));
  wire  [7:0] rom_addr = (n_cur < 9'(HALF)) ? n_cur[7:0] : 8'((FRAME_LEN - 1) - n_cur);

  always_ff @(posedge clk) begin
    if (!rst_n) n_cnt <= '0;
    else if (in_valid) n_cnt <= n_cur + 9'd1;
  end

  logic [15:0] w_rom;
  rom_sync #(.WIDTH(16), .DEPTH(HALF), .INIT_FILE(INIT_FILE)) u_rom (
    .clk(clk), .en(in_valid), .addr(rom_addr[$clog2(HALF)-1:0]), .dout(w_rom)
  );

  // ---- tahap 1: data dan bendera disejajarkan dengan keluaran ROM ----------------------------------
  logic                s1_valid, s1_first, s1_last;
  logic signed [IN_W-1:0] s1_data;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      s1_valid <= 1'b0;
    end else begin
      s1_valid <= in_valid;
      if (in_valid) begin
        s1_data  <= in_data;
        s1_first <= (n_cur == 9'd0);
        s1_last  <= s0_last;
      end
    end
  end

  // ---- tahap 2: perkalian --------------------------------------------------------------------------
  logic                s2_valid, s2_first, s2_last;
  logic signed [PW-1:0] s2_prod;
  wire signed [16:0]   w_s = {1'b0, w_rom};
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      s2_valid <= 1'b0;
    end else begin
      s2_valid <= s1_valid;
      if (s1_valid) begin
        s2_prod  <= s1_data * w_s;
        s2_first <= s1_first;
        s2_last  <= s1_last;
      end
    end
  end

  // ---- tahap 3: pembulatan + saturasi; nol-pad setelah sampel ke-400 ---------------------------------
  localparam int EW = PW + 1;
  wire signed [EW-1:0] prod_e = EW'(s2_prod);
  wire signed [EW-1:0] rnd_v  = (prod_e + (EW'(1) <<< (SH - 1))) >>> SH;
  localparam logic signed [EW-1:0] MAXW = (EW'(1) <<< (W - 1)) - 1;
  localparam logic signed [EW-1:0] MINW = -(EW'(1) <<< (W - 1));

  logic        pad_active;
  logic [7:0]  pad_cnt;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      out_valid  <= 1'b0;
      out_first  <= 1'b0;
      out_last   <= 1'b0;
      pad_active <= 1'b0;
      pad_cnt    <= '0;
    end else begin
      out_valid <= 1'b0;
      out_first <= 1'b0;
      out_last  <= 1'b0;
      if (s2_valid) begin
        out_valid <= 1'b1;
        out_first <= s2_first;
        out_data  <= (rnd_v > MAXW) ? MAXW[W-1:0] : (rnd_v < MINW) ? MINW[W-1:0] : rnd_v[W-1:0];
        if (s2_last) begin
          pad_active <= 1'b1;
          pad_cnt    <= '0;
        end
      end else if (pad_active) begin
        out_valid <= 1'b1;
        out_data  <= '0;
        pad_cnt   <= pad_cnt + 8'd1;
        if (pad_cnt == 8'(NPAD - 1)) begin
          out_last   <= 1'b1;
          pad_active <= 1'b0;
        end
      end
    end
  end
endmodule
