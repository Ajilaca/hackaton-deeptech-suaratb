// i2s_rx: penerima I2S (FPGA = master) untuk mikrofon INMP441, satu kanal.
// Fungsi    : membangkitkan BCLK = clk/CLK_PER_BCLK dan WS (64 BCLK/bingkai), mengambil kata 24 bit
//             MSB-first (tertunda 1 BCLK dari tepi WS), mengeluarkan OUT_BITS teratas (pemotongan = lantai).
// Parameter : CLK_PER_BCLK (96 -> BCLK 1,024 MHz @98,304 MHz; fs = BCLK/64 = 16 kHz), DATA_BITS=24,
//             OUT_BITS=16, RIGHT_CH (0 = slot WS rendah/kiri)
// Antarmuka : en (jalankan BCLK/WS), sd (data mikrofon), bclk, ws, sample (bertanda), sample_valid (1 siklus)
// Latensi   : sample_valid 1 siklus setelah tepi naik BCLK yang mengambil bit terakhir
// Golden    : tidak ada blok golden; kontrak = sample = word24[23:8] (golden/gen/gen_vectors.py: i2s_rx)
module i2s_rx #(
  parameter int CLK_PER_BCLK = 96,
  parameter int DATA_BITS    = 24,
  parameter int OUT_BITS     = 16,
  parameter bit RIGHT_CH     = 1'b0
) (
  input  logic                        clk,
  input  logic                        rst_n,
  input  logic                        en,
  output logic                        bclk,
  output logic                        ws,
  input  logic                        sd,
  output logic signed [OUT_BITS-1:0]  sample,
  output logic                        sample_valid
);
  localparam int HALF = CLK_PER_BCLK / 2;

  logic [$clog2(HALF)-1:0]  div;
  logic                     bclk_q;
  logic                     rise, fall;
  logic [5:0]               cnt;          // jumlah tepi turun sejak awal bingkai (0..63)
  logic [DATA_BITS-1:0]     shift;
  logic                     sd_m, sd_s;

  assign bclk = bclk_q;

  // pembagi clock -> BCLK dan denyut tepi
  always_ff @(posedge clk) begin
    if (!rst_n || !en) begin
      div    <= '0;
      bclk_q <= 1'b0;
      rise   <= 1'b0;
      fall   <= 1'b0;
    end else begin
      rise <= 1'b0;
      fall <= 1'b0;
      if (div == HALF - 1) begin
        div    <= '0;
        bclk_q <= ~bclk_q;
        if (bclk_q) fall <= 1'b1;
        else        rise <= 1'b1;
      end else begin
        div <= div + 1'b1;
      end
    end
  end

  // sinkronisasi sd (2 FF)
  always_ff @(posedge clk) begin
    sd_m <= sd;
    sd_s <= sd_m;
  end

  // WS berubah pada tepi turun BCLK; cnt[5] = 0 untuk slot kiri, 1 untuk kanan
  wire [5:0] cnt_n = cnt + 6'd1;

  always_ff @(posedge clk) begin
    if (!rst_n || !en) begin
      cnt <= '0;
      ws  <= 1'b0;
    end else if (fall) begin
      cnt <= cnt_n;
      ws  <= cnt_n[5];
    end
  end

  wire [DATA_BITS-1:0] word_next = {shift[DATA_BITS-2:0], sd_s};

  // pengambilan: bit ke-k (1..DATA_BITS) pada tepi naik saat cnt[4:0] == k dan slot sesuai
  always_ff @(posedge clk) begin
    sample_valid <= 1'b0;
    if (!rst_n) begin
      shift  <= '0;
      sample <= '0;
    end else if (rise) begin
      if ((cnt[5] == RIGHT_CH) && (cnt[4:0] >= 5'd1) && (cnt[4:0] <= 5'(DATA_BITS))) begin
        shift <= word_next;
        if (cnt[4:0] == 5'(DATA_BITS)) begin
          sample       <= word_next[DATA_BITS-1 -: OUT_BITS];
          sample_valid <= 1'b1;
        end
      end
    end
  end
endmodule
