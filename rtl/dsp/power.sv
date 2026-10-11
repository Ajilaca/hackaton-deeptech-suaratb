// power: spektrum daya P[k] = re^2 + im^2 (tanpa penskalaan), lebar 2W bit tak bertanda (terpakai 2W-1 bit).
// Fungsi    : dua pengali bertanda W x W (re*re, im*im) lalu penjumlah; 1 sampel/siklus.
// Parameter : W
// Antarmuka : in_valid/in_re/in_im/in_idx/in_last -> out_valid/out_p/out_idx/out_last (idx dan last diteruskan)
// Latensi   : 2 siklus
// Golden    : golden/blocks/power.py : power(re, im)
module power #(
  parameter int W = suaratb_pkg::FFT_W
) (
  input  logic                clk,
  input  logic                rst_n,
  input  logic                in_valid,
  input  logic signed [W-1:0] in_re,
  input  logic signed [W-1:0] in_im,
  input  logic [8:0]          in_idx,
  input  logic                in_last,
  output logic                out_valid,
  output logic [2*W-1:0]      out_p,
  output logic [8:0]          out_idx,
  output logic                out_last
);
  logic                 v1;
  logic [8:0]           idx1;
  logic                 last1;
  logic signed [2*W-1:0] rr, ii;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      v1        <= 1'b0;
      out_valid <= 1'b0;
      out_last  <= 1'b0;
    end else begin
      v1        <= in_valid;
      out_valid <= v1;
      out_last  <= v1 && last1;
    end
  end

  always_ff @(posedge clk) begin
    if (in_valid) begin
      rr    <= in_re * in_re;
      ii    <= in_im * in_im;
      idx1  <= in_idx;
      last1 <= in_last;
    end
    if (v1) begin
      out_p   <= rr + ii;           // jumlah dua kuadrat selalu >= 0 dan < 2^(2W-1)
      out_idx <= idx1;
    end
  end
endmodule
