// requant: requantisasi berurutan P lane hasil pe_array dan penulisan ke buffer tujuan.
// Fungsi    : untuk lane l = 0..lanes-1 (1 lane/siklus): y = (x*M0 + 2^(shift-1)) >>> shift ;
//             final=0: ReLU + saturasi [0,127] (int8) ; final=1: saturasi int16 tanpa ReLU.
//             Alamat tulis = base + l*stride (kanal keluaran g*P+l, stride = Ho*Wo).
// Parameter : P, ACC_W, AW
// Antarmuka : start (1 siklus) + res/lanes/base/stride/m0/shift/final -> we/waddr/wdata, done (1 siklus setelah tulis terakhir)
// Latensi   : 4 siklus dari lane pertama ke penulisan
// Golden    : golden/blocks/cnn.py : requant(acc, m0, shift, relu, out_bits)
module requant #(
  parameter int P     = cnn_cfg_pkg::P,
  parameter int ACC_W = suaratb_pkg::ACC_BITS,
  parameter int AW    = 13
) (
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    start,
  input  logic [P*ACC_W-1:0]      res,
  input  logic [7:0]              lanes,
  input  logic [AW-1:0]           base,
  input  logic [AW-1:0]           stride,
  input  logic [15:0]             m0,
  input  logic [4:0]              shift,
  input  logic                    final_l,
  output logic                    we,
  output logic [AW-1:0]           waddr,
  output logic signed [15:0]      wdata,
  output logic                    done
);
  localparam int MW = ACC_W + 17;

  logic          run;
  logic [7:0]    l_cnt;
  logic [AW-1:0] a_cur;

  // tahap 1: pilih lane
  logic                     v1, last1;
  logic signed [ACC_W-1:0]  x1;
  logic [AW-1:0]            a1;
  // tahap 2: kali
  logic                     v2, last2;
  logic signed [MW-1:0]     m2;
  logic [AW-1:0]            a2;
  // tahap 3: bulatkan + geser
  logic                     v3, last3;
  logic signed [MW-1:0]     y3;
  logic [AW-1:0]            a3;

  wire signed [16:0] m0_s = {1'b0, m0[15:0]};
  wire signed [MW-1:0] rnd = (MW'(1) <<< (shift - 5'd1));

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      run <= 1'b0; v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; we <= 1'b0; done <= 1'b0;
      l_cnt <= '0; a_cur <= '0;
    end else begin
      done <= 1'b0;
      we   <= 1'b0;
      // tahap 0: penerbit lane
      v1 <= 1'b0;
      if (start) begin
        run   <= 1'b1;
        l_cnt <= '0;
        a_cur <= base;
      end else if (run) begin
        v1    <= 1'b1;
        x1    <= res[ACC_W*l_cnt +: ACC_W];
        a1    <= a_cur;
        last1 <= (l_cnt == lanes - 8'd1);
        a_cur <= a_cur + stride;
        l_cnt <= l_cnt + 8'd1;
        if (l_cnt == lanes - 8'd1) run <= 1'b0;
      end
      // tahap 2
      v2 <= v1;
      if (v1) begin
        m2    <= x1 * m0_s;
        a2    <= a1;
        last2 <= last1;
      end
      // tahap 3
      v3 <= v2;
      if (v2) begin
        y3    <= (m2 + rnd) >>> shift;
        a3    <= a2;
        last3 <= last2;
      end
      // tahap 4: saturasi dan tulis
      if (v3) begin
        we    <= 1'b1;
        waddr <= a3;
        done  <= last3;
        if (final_l) wdata <= (y3 > 40'sd32767) ? 16'sd32767 : (y3 < -40'sd32768) ? -16'sd32768 : y3[15:0];
        else         wdata <= (y3 < 0) ? 16'sd0 : (y3 > 40'sd127) ? 16'sd127 : y3[15:0];
      end
    end
  end
endmodule
