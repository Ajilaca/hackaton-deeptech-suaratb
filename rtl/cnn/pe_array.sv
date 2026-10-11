// pe_array: P lane MAC 8x8 bertanda, akumulator ACC_W bit, bias awal, maks antar-posisi pool, bank hasil.
// Fungsi    : satu aktivasi `act` (dibroadcast) x P bobot (satu per lane = satu kanal keluaran). Alur per token:
//               s2: prod[l] = act * w[l]                          (token dengan first: bias_q <= bias)
//               s3: acc[l]  = (first ? bias_q[l] : acc[l]) + prod[l]
//               s4: bila lastpos: n = posfirst ? acc : max(pmax, acc); pmax = n; bila pixlast: res = n, res_valid
//             MaxPool dilakukan pada akumulator sebelum requant (identik golden: requant+ReLU monoton).
// Parameter : P, ACC_W, BIAS_W
// Antarmuka : in_valid/in_first/in_lastpos/in_posfirst/in_pixlast, in_act (int8, 0 bila di luar batas), in_w (P*8 bit),
//             in_bias (P*BIAS_W bit, valid saat in_first) -> res (P*ACC_W), res_valid (1 siklus)
// Latensi   : 5 siklus dari token ke res_valid (register masukan; pengali dipecah dua separuh; banding pool terpisah)
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
  // tahap 1: masukan didaftarkan (aktivasi dan bobot datang dari keluaran BRAM/ROM sinkron)
  logic                         v2, first2, lastpos2, posfirst2, pixlast2;
  logic signed [7:0]            act_q;
  logic [P*8-1:0]               w_q;
  logic                         v3, first3, lastpos3, posfirst3, pixlast3;
  logic                         v4, lastpos4, posfirst4, pixlast4;
  logic                         v5, lastpos5, posfirst5, pixlast5;
  logic [P-1:0]                 gt_q;
  logic signed [ACC_W-1:0]      accf [0:P-1];
  logic [P*BIAS_W-1:0]          bias_q;
  logic signed [11:0]           prod_h [0:P-1];   // act * w[7:4] (bertanda)
  logic signed [12:0]           prod_l [0:P-1];   // act * {0, w[3:0]}
  logic signed [ACC_W-1:0]      acc  [0:P-1];
  logic signed [ACC_W-1:0]      pmax [0:P-1];

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0; v5 <= 1'b0; res_valid <= 1'b0;
    end else begin
      v2        <= in_valid;
      v3        <= v2;
      v4        <= v3;
      v5        <= v4;
      res_valid <= v5 && lastpos5 && pixlast5;
    end
    // s1: daftarkan masukan; bias ditangkap pada token first (tetap valid sampai token first berikutnya >= 9 siklus)
    act_q <= in_act;                              // bebas-jalan (nilai valid hanya dipakai bersama bendera valid)
    w_q   <= in_w;
    if (in_valid) begin
      first2    <= in_first;
      lastpos2  <= in_lastpos;
      posfirst2 <= in_posfirst;
      pixlast2  <= in_pixlast;
      if (in_first) bias_q <= in_bias;
    end
    // s2: produk
    if (v2) begin
      first3    <= first2;
      lastpos3  <= lastpos2;
      posfirst3 <= posfirst2;
      pixlast3  <= pixlast2;
    end
    // s3 -> s4 -> s5 bendera
    if (v3) begin
      lastpos4  <= lastpos3;
      posfirst4 <= posfirst3;
      pixlast4  <= pixlast3;
    end
    if (v4) begin
      lastpos5  <= lastpos4;
      posfirst5 <= posfirst4;
      pixlast5  <= pixlast4;
    end
  end

  genvar l;
  generate
    for (l = 0; l < P; l++) begin : g_lane
      wire signed [7:0] w_l = w_q[8*l +: 8];
      wire signed [3:0] w_h = w_l[7:4];
      wire        [3:0] w_lo = w_l[3:0];
      wire signed [ACC_W-1:0] bias_l = bias_q[BIAS_W*l +: BIAS_W];     // BIAS_W <= ACC_W (24)
      wire signed [ACC_W-1:0] prod_w = (ACC_W'(prod_h[l]) <<< 4) + ACC_W'(prod_l[l]);
      wire signed [ACC_W-1:0] new_max = posfirst5 ? accf[l] : (gt_q[l] ? accf[l] : pmax[l]);

      always_ff @(posedge clk) begin
        // s2: dua produk separuh (8x4 dan 8x5)
        prod_h[l] <= act_q * w_h;
        prod_l[l] <= act_q * $signed({1'b0, w_lo});
        // s3: akumulasi
        if (v3) acc[l] <= (first3 ? bias_l : acc[l]) + prod_w;
        // s4: banding dengan maks sebelumnya; salin akumulator final
        accf[l] <= acc[l];
        gt_q[l] <= (acc[l] > pmax[l]);
        // s5: maks posisi pool
        if (v5 && lastpos5) begin
          pmax[l] <= new_max;
          if (pixlast5) res[ACC_W*l +: ACC_W] <= new_max;
        end
      end
    end
  endgenerate
endmodule
