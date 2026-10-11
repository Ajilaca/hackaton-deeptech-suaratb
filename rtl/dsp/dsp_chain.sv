// dsp_chain: rantai DSP satu frame: window -> fft512 -> power -> mel -> log_q -> quant.
// Fungsi    : frame 400 sampel (16+IN_FRAC bit, dari frame_buf) -> 40 nilai int8 (urutan band 0..39).
// Antarmuka : in_valid/in_data/in_first (400 sampel berurutan); out_valid/out_q/out_idx/out_last (40 nilai);
//             busy (fft512 sedang menghitung/mengeluarkan), n_sat (saturasi FFT frame terakhir, valid saat fft_done)
// Latensi   : ~7-8 rb siklus per frame (FFT dominan); frame berikutnya baru boleh masuk setelah out_last
//             (jarak frame audio 160 sampel = 983 rb siklus @98,304 MHz, jauh lebih longgar).
// Golden    : golden/blocks/frontend.py : features_q(x)[f] (baris frame f)
module dsp_chain #(
  parameter int W        = suaratb_pkg::FFT_W,
  parameter int IN_FRAC  = suaratb_pkg::PREEMPH_FRAC,
  parameter INIT_WIN = "rtl/mem/window_hamming.mem",
  parameter INIT_COS = "rtl/mem/twiddle_cos.mem",
  parameter INIT_SIN = "rtl/mem/twiddle_sin.mem",
  parameter INIT_MEL = "rtl/mem/mel_rom.mem",
  parameter INIT_LOG = "rtl/mem/log_lut.mem"
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         in_valid,
  input  logic signed [16+IN_FRAC-1:0] in_data,
  input  logic                         in_first,
  output logic                         out_valid,
  output logic signed [7:0]            out_q,
  output logic [5:0]                   out_idx,
  output logic                         out_last,
  output logic                         fft_done,
  output logic [31:0]                  n_sat
);
  localparam int LOGW = $clog2(suaratb_pkg::LOG_IN_BITS) + suaratb_pkg::LOG_FRAC;

  // window
  logic                w_valid, w_first, w_last;
  logic signed [W-1:0] w_data;
  window #(.W(W), .IN_FRAC(IN_FRAC), .INIT_FILE(INIT_WIN)) u_win (
    .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_data(in_data), .in_first(in_first),
    .out_valid(w_valid), .out_data(w_data), .out_first(w_first), .out_last(w_last)
  );

  // fft512
  logic                f_ready, f_valid, f_last;
  logic signed [W-1:0] f_re, f_im;
  logic [8:0]          f_idx;
  fft512 #(.W(W), .INIT_COS(INIT_COS), .INIT_SIN(INIT_SIN)) u_fft (
    .clk(clk), .rst_n(rst_n), .in_valid(w_valid), .in_data(w_data), .in_first(w_first), .in_ready(f_ready),
    .out_valid(f_valid), .out_re(f_re), .out_im(f_im), .out_idx(f_idx), .out_last(f_last),
    .done(fft_done), .n_sat(n_sat)
  );

  // power
  logic                p_valid, p_last;
  logic [2*W-1:0]      p_data;
  logic [8:0]          p_idx;
  power #(.W(W)) u_pow (
    .clk(clk), .rst_n(rst_n), .in_valid(f_valid), .in_re(f_re), .in_im(f_im), .in_idx(f_idx), .in_last(f_last),
    .out_valid(p_valid), .out_p(p_data), .out_idx(p_idx), .out_last(p_last)
  );

  // mel
  logic                         m_valid, m_last;
  logic [suaratb_pkg::LOG_IN_BITS-1:0] m_e;
  logic [5:0]                   m_idx;
  mel #(.P_W(2*W), .INIT_FILE(INIT_MEL)) u_mel (
    .clk(clk), .rst_n(rst_n), .in_valid(p_valid), .in_p(p_data), .in_first(p_idx == 9'd0), .in_last(p_last),
    .out_valid(m_valid), .out_e(m_e), .out_idx(m_idx), .out_last(m_last)
  );

  // log2
  logic                l_valid, l_last;
  logic [LOGW-1:0]     l_data;
  logic [5:0]          l_idx;
  log_q #(.INIT_FILE(INIT_LOG)) u_log (
    .clk(clk), .rst_n(rst_n), .in_valid(m_valid), .in_e(m_e), .in_idx(m_idx), .in_last(m_last),
    .out_valid(l_valid), .out_l(l_data), .out_idx(l_idx), .out_last(l_last)
  );

  // kuantisasi
  quant u_q (
    .clk(clk), .rst_n(rst_n), .in_valid(l_valid), .in_l(l_data), .in_idx(l_idx), .in_last(l_last),
    .out_valid(out_valid), .out_q(out_q), .out_idx(out_idx), .out_last(out_last)
  );
endmodule
