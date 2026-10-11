// preemph: pre-emphasis y[n] = x[n] - a*x[n-1], a = COEF/2^15 (0,97).
// Fungsi    : dua keluaran dari satu perkalian a*x[n-1]:
//               out_wide = sat_(16+FRAC)( (x<<FRAC) - round(a*xp / 2^(15-FRAC)) )   -> jalur FFT (K-08)
//               out_s16  = sat_16( x - round(a*xp / 2^15) )                         -> jalur detektor energi
//             round(v/2^s) = (v + 2^(s-1)) >>> s (tambah setengah lalu geser aritmetik).
// Parameter : FRAC (bit pecahan keluaran lebar, = FFT_GUARD), COEF (Q15)
// Antarmuka : in_valid/in_sample (int16) -> out_valid/out_wide/out_s16 (stream valid saja)
// Latensi   : 4 siklus (kali; bulatkan; kurangi; saturasi). Keadaan awal x[-1] = 0 (reset).
// Golden    : golden/blocks/preemph.py : preemph(x, frac=FRAC) dan preemph(x) (frac=0)
module preemph #(
  parameter int FRAC = suaratb_pkg::PREEMPH_FRAC,
  parameter int COEF = suaratb_pkg::PREEMPH_Q15
) (
  input  logic                          clk,
  input  logic                          rst_n,
  input  logic                          in_valid,
  input  logic signed [15:0]            in_sample,
  output logic                          out_valid,
  output logic signed [16+FRAC-1:0]     out_wide,
  output logic signed [15:0]            out_s16
);
  localparam int OW   = 16 + FRAC;
  localparam int SH_W = 15 - FRAC;                       // geser pembulatan jalur lebar (>= 1)
  localparam logic signed [16:0] COEF_S = 17'(COEF);

  logic signed [15:0] x_prev;
  logic signed [15:0] x1;
  logic signed [32:0] prod1;
  logic               v1;

  // tahap 1: perkalian a * x[n-1]
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      x_prev <= '0;
      v1     <= 1'b0;
    end else begin
      v1 <= in_valid;
      if (in_valid) begin
        x1     <= in_sample;
        prod1  <= COEF_S * x_prev;
        x_prev <= in_sample;
      end
    end
  end

  // tahap 2: pembulatan hasil kali (tambah setengah + geser aritmetik)
  logic                v2, v3;
  logic signed [33:0]  r_wide_q, r_16_q, x_e_q;
  logic signed [33:0]  y_wide_q, y_16_q;

  wire signed [33:0] prod_e   = prod1;
  wire signed [33:0] x_e      = {{18{x1[15]}}, x1};
  wire signed [33:0] r_wide   = (prod_e + (34'sd1 <<< (SH_W - 1))) >>> SH_W;
  wire signed [33:0] r_16     = (prod_e + 34'sd16384) >>> 15;

  localparam logic signed [33:0] MAXW = (34'sd1 <<< (OW - 1)) - 34'sd1;
  localparam logic signed [33:0] MINW = -(34'sd1 <<< (OW - 1));
  localparam logic signed [33:0] MAX16 = 34'sd32767;
  localparam logic signed [33:0] MIN16 = -34'sd32768;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      v2        <= 1'b0;
      v3        <= 1'b0;
      out_valid <= 1'b0;
    end else begin
      v2        <= v1;
      v3        <= v2;
      out_valid <= v3;
    end
    // tahap 2: bulatkan
    if (v1) begin
      r_wide_q <= r_wide;
      r_16_q   <= r_16;
      x_e_q    <= x_e;
    end
    // tahap 3: kurangi
    if (v2) begin
      y_wide_q <= (x_e_q <<< FRAC) - r_wide_q;
      y_16_q   <= x_e_q - r_16_q;
    end
    // tahap 4: saturasi
    if (v3) begin
      out_wide <= (y_wide_q > MAXW) ? OW'(MAXW) : (y_wide_q < MINW) ? OW'(MINW) : y_wide_q[OW-1:0];
      out_s16  <= (y_16_q > MAX16) ? 16'(MAX16) : (y_16_q < MIN16) ? 16'(MIN16) : y_16_q[15:0];
    end
  end
endmodule
