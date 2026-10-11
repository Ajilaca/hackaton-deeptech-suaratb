// log_q: log2 titik tetap Q(clog2(IN_BITS)).8 dengan leading-one + 8 bit mantissa + ROM 256 entri.
// Fungsi    : p = posisi leading-one (0..IN_BITS-1); m = 8 bit tepat di bawahnya
//             (p >= 8: m = (e >> (p-8)) & 255 ; p < 8: m = (e << (8-p)) & 255);
//             L = (p << 8) + LUT[m]; e = 0 -> L = 0.
// Parameter : IN_BITS (LOG_IN_BITS, 58), LUT_BITS (8), FRAC (8), INIT_FILE
// Antarmuka : in_valid/in_e/in_idx/in_last -> out_valid/out_l/out_idx/out_last (idx dan last diteruskan)
// Latensi   : 3 siklus; 1 sampel/siklus
// Golden    : golden/blocks/log_q.py : log2_q(e), log_lut()
module log_q #(
  parameter int    IN_BITS   = suaratb_pkg::LOG_IN_BITS,
  parameter int    LUT_BITS  = suaratb_pkg::LOG_LUT_BITS,
  parameter int    FRAC      = suaratb_pkg::LOG_FRAC,
  parameter INIT_FILE = "rtl/mem/log_lut.mem"
) (
  input  logic                          clk,
  input  logic                          rst_n,
  input  logic                          in_valid,
  input  logic [IN_BITS-1:0]            in_e,
  input  logic [5:0]                    in_idx,
  input  logic                          in_last,
  output logic                          out_valid,
  output logic [$clog2(IN_BITS)+FRAC-1:0] out_l,
  output logic [5:0]                    out_idx,
  output logic                          out_last
);
  localparam int PW = $clog2(IN_BITS);          // bit posisi leading-one (6)
  localparam int OW = PW + FRAC;                // 14

  // tahap 1: priority encoder
  logic [IN_BITS-1:0] e1;
  logic [PW-1:0]      p1;
  logic               z1, v1, last1;
  logic [5:0]         idx1;

  function automatic logic [PW-1:0] lead_pos(input logic [IN_BITS-1:0] v);
    lead_pos = '0;
    for (int i = 0; i < IN_BITS; i++) if (v[i]) lead_pos = PW'(i);
  endfunction

  always_ff @(posedge clk) begin
    if (!rst_n) v1 <= 1'b0;
    else        v1 <= in_valid;
    if (in_valid) begin
      e1    <= in_e;
      p1    <= lead_pos(in_e);
      z1    <= (in_e == '0);
      idx1  <= in_idx;
      last1 <= in_last;
    end
  end

  // tahap 2: barrel shift -> mantissa 8 bit -> alamat ROM
  logic [IN_BITS-1:0] sh_l, sh_r;
  logic [LUT_BITS-1:0] m2;
  always_comb begin
    sh_r = e1 >> ((p1 >= PW'(LUT_BITS)) ? (p1 - PW'(LUT_BITS)) : '0);
    sh_l = e1 << ((p1 <  PW'(LUT_BITS)) ? (PW'(LUT_BITS) - p1) : '0);
    m2   = (p1 >= PW'(LUT_BITS)) ? sh_r[LUT_BITS-1:0] : sh_l[LUT_BITS-1:0];
  end

  logic [FRAC-1:0] lut;
  rom_sync #(.WIDTH(FRAC), .DEPTH(1 << LUT_BITS), .INIT_FILE(INIT_FILE)) u_lut (
    .clk(clk), .en(v1), .addr(m2), .dout(lut)
  );

  logic [PW-1:0] p2;
  logic          z2, v2, last2;
  logic [5:0]    idx2;
  always_ff @(posedge clk) begin
    if (!rst_n) v2 <= 1'b0;
    else        v2 <= v1;
    if (v1) begin
      p2    <= p1;
      z2    <= z1;
      idx2  <= idx1;
      last2 <= last1;
    end
  end

  // tahap 3: penjumlahan
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      out_valid <= 1'b0;
      out_last  <= 1'b0;
    end else begin
      out_valid <= v2;
      out_last  <= v2 && last2;
    end
    if (v2) begin
      out_l   <= z2 ? '0 : OW'({p2, {FRAC{1'b0}}}) + OW'(lut);
      out_idx <= idx2;
    end
  end
endmodule
