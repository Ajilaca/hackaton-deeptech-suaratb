// quant: kuantisasi fitur log-mel: q = sat8( round((L - OFFSET) / 2^SHIFT) ), round = (v + 2^(SHIFT-1)) >>> SHIFT.
// Parameter : IN_W (lebar L), OFFSET (FEAT_OFFSET), SHIFT (FEAT_SHIFT)
// Antarmuka : in_valid/in_l/in_idx/in_last -> out_valid/out_q (int8)/out_idx/out_last
// Latensi   : 1 siklus
// Golden    : golden/blocks/quant.py : quant_feat(L, offset, shift)
module quant #(
  parameter int IN_W   = $clog2(suaratb_pkg::LOG_IN_BITS) + suaratb_pkg::LOG_FRAC,
  parameter int OFFSET = suaratb_pkg::FEAT_OFFSET,
  parameter int SHIFT  = suaratb_pkg::FEAT_SHIFT
) (
  input  logic                clk,
  input  logic                rst_n,
  input  logic                in_valid,
  input  logic [IN_W-1:0]     in_l,
  input  logic [5:0]          in_idx,
  input  logic                in_last,
  output logic                out_valid,
  output logic signed [7:0]   out_q,
  output logic [5:0]          out_idx,
  output logic                out_last
);
  localparam int VW = IN_W + 3;
  wire signed [VW-1:0] v   = $signed({3'b000, in_l}) - VW'(OFFSET);
  wire signed [VW-1:0] r   = (SHIFT == 0) ? v : ((v + (VW'(1) <<< (SHIFT - 1 > 0 ? SHIFT - 1 : 0))) >>> SHIFT);
  localparam logic signed [VW-1:0] HI = VW'(127);
  localparam logic signed [VW-1:0] LO = -VW'(128);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      out_valid <= 1'b0;
      out_last  <= 1'b0;
    end else begin
      out_valid <= in_valid;
      out_last  <= in_valid && in_last;
    end
    if (in_valid) begin
      out_q   <= (r > HI) ? 8'sd127 : (r < LO) ? -8'sd128 : r[7:0];
      out_idx <= in_idx;
    end
  end
endmodule
