// requant: requantisasi berurutan P lane hasil pe_array dan penulisan ke buffer tujuan.
// Fungsi    : untuk lane l = 0..lanes-1 (1 lane/siklus): y = (x*M0 + 2^(shift-1)) >>> shift ;
//             final=0: ReLU + saturasi [0,127] (int8) ; final=1: saturasi int16 tanpa ReLU.
//             Alamat tulis = base + l*stride (kanal keluaran g*P+l, stride = Ho*Wo).
// Parameter : P, ACC_W, AW
// Antarmuka : start (1 siklus) + res/lanes/base/stride/m0/shift/final -> we/waddr/wdata, done (1 siklus setelah tulis terakhir)
// Latensi   : 6 siklus dari lane pertama ke penulisan (kali, jumlah, geser, saturasi dipisah per tahap)
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
  logic [P*ACC_W-1:0] sr;

  // tahap 1: pilih lane
  logic                     v1, last1;
  logic signed [ACC_W-1:0]  x1;
  logic [AW-1:0]            a1;
  // tahap 2: kali (dua register agar blok DSP dapat memakai register M dan P)
  logic                     v2, last2, v2b, last2b;
  logic signed [MW-1:0]     m2, m2b;
  logic [AW-1:0]            a2, a2b;
  // tahap 3: + pembulatan
  logic                     v3, last3;
  logic signed [MW-1:0]     s3;
  logic [AW-1:0]            a3;
  // tahap 4: geser
  logic                     v4, last4;
  logic signed [MW-1:0]     y4;
  logic [AW-1:0]            a4;
  logic signed [MW-1:0]     rnd_q;
  logic [4:0]               shift_q;

  wire signed [16:0] m0_s = {1'b0, m0[15:0]};

  always_ff @(posedge clk) begin
    // konstanta layer didaftarkan (tidak berubah selama satu layer)
    shift_q <= shift;
    rnd_q   <= (MW'(1) <<< (shift - 5'd1));
    if (!rst_n) begin
      run <= 1'b0; v1 <= 1'b0; v2 <= 1'b0; v2b <= 1'b0; v3 <= 1'b0; v4 <= 1'b0; we <= 1'b0; done <= 1'b0;
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
        sr    <= res;                  // geser keluar satu lane per siklus (tanpa mux P:1)
      end else if (run) begin
        v1    <= 1'b1;
        x1    <= sr[ACC_W-1:0];
        sr    <= sr >> ACC_W;
        a1    <= a_cur;
        last1 <= (l_cnt == lanes - 8'd1);
        a_cur <= a_cur + stride;
        l_cnt <= l_cnt + 8'd1;
        if (l_cnt == lanes - 8'd1) run <= 1'b0;
      end
      // tahap 2: kali
      v2 <= v1;
      if (v1) begin
        m2    <= x1 * m0_s;
        a2    <= a1;
        last2 <= last1;
      end
      v2b <= v2;
      if (v2) begin
        m2b    <= m2;
        a2b    <= a2;
        last2b <= last2;
      end
      // tahap 3: tambah pembulatan
      v3 <= v2b;
      if (v2b) begin
        s3    <= m2b + rnd_q;
        a3    <= a2b;
        last3 <= last2b;
      end
      // tahap 4: geser aritmetik
      v4 <= v3;
      if (v3) begin
        y4    <= s3 >>> shift_q;
        a4    <= a3;
        last4 <= last3;
      end
      // tahap 5: saturasi dan tulis
      if (v4) begin
        we    <= 1'b1;
        waddr <= a4;
        done  <= last4;
        if (final_l) wdata <= (y4 > 40'sd32767) ? 16'sd32767 : (y4 < -40'sd32768) ? -16'sd32768 : y4[15:0];
        else         wdata <= (y4 < 0) ? 16'sd0 : (y4 > 40'sd127) ? 16'sd127 : y4[15:0];
      end
    end
  end
endmodule
