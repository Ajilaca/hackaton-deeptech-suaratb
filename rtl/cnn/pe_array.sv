// pe_array: P lane MAC 8x8 bertanda, akumulator ACC_W bit, bias awal, maks antar-posisi pool, bank hasil.
// Fungsi    : satu aktivasi `act` (dibroadcast) x P bobot (satu per lane = satu kanal keluaran). Alur per token:
//               s2: prod[l] = act * w[l]                          (token dengan first: bias_q <= bias)
//               s3: acc[l]  = (first ? bias_q[l] : acc[l]) + prod[l]
//               s4: bila lastpos: n = posfirst ? acc : max(pmax, acc); pmax = n; bila pixlast: res = n, res_valid
//             MaxPool dilakukan pada akumulator sebelum requant (identik golden: requant+ReLU monoton).
// Parameter : P, ACC_W, BIAS_W
// Antarmuka : in_valid/in_first/in_lastpos/in_posfirst/in_pixlast, in_act (int8, 0 bila di luar batas), in_w (P*8 bit),
//             in_bias (P*BIAS_W bit, valid saat in_first) -> res (P*ACC_W), res_valid (1 siklus)
// Latensi   : 3 siklus dari token ke res_valid
// Golden    : golden/blocks/cnn.py : conv3x3 + maxpool2 (urutan requant/ReLU dipindah setelah maks)
module pe_array #(
  parameter int P      = cnn_cfg_pkg::P,
  parameter int ACC_W  = suaratb_pkg::ACC_BITS,
  parameter int BIAS_W = suaratb_pkg::BIAS_BITS
) (
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    in_valid,
  input  logic                    in_first,
  input  logic                    in_lastpos,
  input  logic                    in_posfirst,
  input  logic                    in_pixlast,
  input  logic signed [7:0]       in_act,
  input  logic [P*8-1:0]          in_w,
  input  logic [P*BIAS_W-1:0]     in_bias,
  output logic [P*ACC_W-1:0]      res,
  output logic                    res_valid
);
  logic                         v3, first3, lastpos3, posfirst3, pixlast3;
  logic                         v4, lastpos4, posfirst4, pixlast4;
  logic [P*BIAS_W-1:0]          bias_q;
  logic signed [15:0]           prod [0:P-1];
  logic signed [ACC_W-1:0]      acc  [0:P-1];
  logic signed [ACC_W-1:0]      pmax [0:P-1];

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      v3 <= 1'b0; v4 <= 1'b0; res_valid <= 1'b0;
    end else begin
      v3        <= in_valid;
      v4        <= v3;
      res_valid <= v4 && lastpos4 && pixlast4;
    end
    // s2: produk + bias
    if (in_valid) begin
      first3    <= in_first;
      lastpos3  <= in_lastpos;
      posfirst3 <= in_posfirst;
      pixlast3  <= in_pixlast;
      if (in_first) bias_q <= in_bias;
    end
    // s3 -> s4 bendera
    if (v3) begin
      lastpos4  <= lastpos3;
      posfirst4 <= posfirst3;
      pixlast4  <= pixlast3;
    end
  end

  genvar l;
  generate
    for (l = 0; l < P; l++) begin : g_lane
      wire signed [7:0] w_l = in_w[8*l +: 8];
      wire signed [ACC_W-1:0] bias_l = bias_q[BIAS_W*l +: BIAS_W];     // BIAS_W <= ACC_W (24)
      wire signed [ACC_W-1:0] new_max = posfirst4 ? acc[l] : ((acc[l] > pmax[l]) ? acc[l] : pmax[l]);

      always_ff @(posedge clk) begin
        if (in_valid) prod[l] <= in_act * w_l;
        if (v3) acc[l] <= (first3 ? bias_l : acc[l]) + ACC_W'(prod[l]);
        if (v4 && lastpos4) begin
          pmax[l] <= new_max;
          if (pixlast4) res[ACC_W*l +: ACC_W] <= new_max;
        end
      end
    end
  endgenerate
endmodule
