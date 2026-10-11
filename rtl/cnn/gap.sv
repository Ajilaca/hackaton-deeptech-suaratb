// gap: global average pooling bilangan bulat: out[c] = (sum_i x[c*HW+i] * RECIP + 2^(FRAC-1)) >> FRAC.
// Fungsi    : membaca CH*HW nilai urut dari buffer sumber (1/siklus, data valid 1 siklus setelah rd_en), menjumlah per
//             kanal, menulis CH hasil int8 ke buffer tujuan alamat c.
// Parameter : CH, HW, RECIP (round(2^FRAC/HW)), FRAC, AW
// Antarmuka : start -> rd_en/rd_addr, rd_data (valid 1 siklus setelah rd_en) -> we/waddr/wdata, done
// Latensi   : CH*HW + ~5 siklus
// Golden    : golden/blocks/cnn.py : gap(x)
module gap #(
  parameter int CH    = cnn_cfg_pkg::GAP_CH,
  parameter int HW    = cnn_cfg_pkg::GAP_HW,
  parameter int RECIP = cnn_cfg_pkg::GAP_RECIP,
  parameter int FRAC  = suaratb_pkg::GAP_RECIP_FRAC,
  parameter int AW    = 13
) (
  input  logic                clk,
  input  logic                rst_n,
  input  logic                start,
  output logic                rd_en,
  output logic [AW-1:0]       rd_addr,
  input  logic signed [7:0]   rd_data,
  output logic                we,
  output logic [AW-1:0]       waddr,
  output logic signed [7:0]   wdata,
  output logic                done
);
  logic [AW-1:0] n_cnt;                       // indeks linear 0..CH*HW-1
  logic [7:0]    i_cnt;                       // posisi dalam kanal
  logic [7:0]    ch_cnt;
  logic          run;

  logic          v1, first1, last1;
  logic [7:0]    ch1;
  localparam int SW = 8 + $clog2(HW) + 2;     // lebar jumlah: 8 bit + log2(HW) + margin
  logic signed [SW-1:0] sum;
  logic signed [SW-1:0] sum_f;               // jumlah kanal (didaftarkan sebelum dikali RECIP)
  logic          vf;
  logic          last_chf;
  logic [7:0]    chf;
  logic          last_allf;
  logic          v2, last_ch2;
  logic [7:0]    ch2;
  logic signed [SW+11-1:0] prod2;
  logic          v3, last_all3;
  logic [7:0]    ch3;
  logic          last_all1, last_all2;

  assign rd_en   = run;
  assign rd_addr = n_cnt;

  wire signed [SW-1:0] s_next = (first1 ? '0 : sum) + SW'($signed(rd_data));

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      run <= 1'b0; v1 <= 1'b0; vf <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; we <= 1'b0; done <= 1'b0;
      n_cnt <= '0; i_cnt <= '0; ch_cnt <= '0;
    end else begin
      we   <= 1'b0;
      done <= 1'b0;
      if (start) begin
        run <= 1'b1; n_cnt <= '0; i_cnt <= '0; ch_cnt <= '0;
      end else if (run) begin
        n_cnt <= n_cnt + 1'b1;
        if (i_cnt == 8'(HW - 1)) begin
          i_cnt  <= '0;
          ch_cnt <= ch_cnt + 8'd1;
          if (ch_cnt == 8'(CH - 1)) run <= 1'b0;
        end else begin
          i_cnt <= i_cnt + 8'd1;
        end
      end
      // tahap 1: data tiba
      v1        <= run;
      first1    <= (i_cnt == 8'd0);
      last1     <= (i_cnt == 8'(HW - 1));
      last_all1 <= (i_cnt == 8'(HW - 1)) && (ch_cnt == 8'(CH - 1));
      ch1       <= ch_cnt;
      // tahap 2: jumlah; saat kanal selesai, jumlah didaftarkan (sum_f)
      vf <= 1'b0;
      if (v1) begin
        sum <= s_next;
        if (last1) begin
          vf        <= 1'b1;
          sum_f     <= s_next;
          chf       <= ch1;
          last_allf <= last_all1;
        end
      end
      // tahap 2b: kali RECIP
      v2 <= 1'b0;
      if (vf) begin
        v2        <= 1'b1;
        prod2     <= sum_f * RECIP;
        ch2       <= chf;
        last_all2 <= last_allf;
      end
      // tahap 3: pembulatan
      v3 <= v2;
      if (v2) begin
        ch3       <= ch2;
        last_all3 <= last_all2;
        wdata     <= 8'((prod2 + (FRAC > 0 ? (1 <<< (FRAC - 1)) : 0)) >>> FRAC);
      end
      // tahap 4: tulis
      if (v3) begin
        we    <= 1'b1;
        waddr <= AW'(ch3);
        done  <= last_all3;
      end
    end
  end
endmodule
