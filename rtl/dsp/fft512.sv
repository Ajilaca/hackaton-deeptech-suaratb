// fft512: FFT 512 titik radix-2 DIT, masukan real (bagian imajiner nol), 1 butterfly, RAM in-place,
//         data W bit bertanda, twiddle Q1.15, skala 1/2 per stage (FFT_STAGE_SHIFT_MASK).
// Fungsi    : LOAD   : 512 sampel masukan berurutan ditulis ke alamat bit-reversed (re=sampel, im=0).
//             COMPUTE: 9 stage x 256 butterfly; tiap butterfly 2 siklus (baca a, baca b), alur 7 tahap, tulis a' lalu b'.
//                      t = b*W (W = cos - j sin), a' = (a*2^15 + t + 2^(s+14)) >>> (15+s), b' = (a*2^15 - t + ...) >>> ...,
//                      s = bit stage ke-st dari FFT_STAGE_SHIFT_MASK; hasil disaturasi ke W bit (n_sat menghitung saturasi).
//             OUTPUT : bin 0..256 (re, im) berurutan tiap OUT_GAP siklus, lalu done.
// Parameter : W, TW_W (16), NFFT (512), STAGE_MASK, DRAIN (siklus kosong antar-stage, >= 10), OUT_GAP (>= 3, default 4),
//             INIT_COS, INIT_SIN (ROM twiddle 256 x 16 bit, golden/tables)
// Antarmuka : in_valid/in_data/in_first (512 sampel berurutan; hanya diterima saat in_ready=1)
//             -> out_valid/out_re/out_im/out_idx/out_last, done (1 siklus setelah bin 256), n_sat (valid saat done)
// Latensi   : 512 (muat) + ~2*2304 + 9*DRAIN + 257*OUT_GAP siklus ~ 6,3 rb siklus pada DRAIN=10, OUT_GAP=4 (analitik)
// Golden    : golden/blocks/fft512.py : fft512(xr, W, stage_shift) (keluaran urutan alami, n_sat)
module fft512 #(
  parameter int W          = suaratb_pkg::FFT_W,
  parameter int TW_W       = 16,
  parameter int NFFT       = suaratb_pkg::NFFT,
  parameter int STAGE_MASK = suaratb_pkg::FFT_STAGE_SHIFT_MASK,
  parameter int DRAIN      = 10,
  parameter int OUT_GAP    = 4,
  parameter INIT_COS = "rtl/mem/twiddle_cos.mem",
  parameter INIT_SIN = "rtl/mem/twiddle_sin.mem"
) (
  input  logic                clk,
  input  logic                rst_n,
  input  logic                in_valid,
  input  logic signed [W-1:0] in_data,
  input  logic                in_first,
  output logic                in_ready,
  output logic                out_valid,
  output logic signed [W-1:0] out_re,
  output logic signed [W-1:0] out_im,
  output logic [8:0]          out_idx,
  output logic                out_last,
  output logic                done,
  output logic [31:0]         n_sat
);
  localparam int AW      = 9;                       // log2(NFFT)
  localparam int STAGES  = 9;
  localparam int NBF     = NFFT / 2;                // 256 butterfly per stage
  localparam int PW      = W + TW_W;                // lebar hasil kali
  localparam int AW_SUM  = PW + 2;                  // lebar jumlah (a*2^15 + t + rnd)
  localparam int NBINS   = NFFT / 2 + 1;            // 257

  typedef enum logic [2:0] {S_LOAD, S_COMP, S_STALL, S_OUT, S_DONE} state_t;
  state_t state;

  // ---- RAM data {im, re} -------------------------------------------------------------------------
  logic              ram_we;
  logic [AW-1:0]     ram_waddr;
  logic [2*W-1:0]    ram_wdata;
  logic              ram_re;
  logic [AW-1:0]     ram_raddr;
  logic [2*W-1:0]    ram_rdata;
  ram_sdp #(.WIDTH(2*W), .DEPTH(NFFT)) u_ram (
    .clk(clk), .we(ram_we), .waddr(ram_waddr), .wdata(ram_wdata),
    .re(ram_re), .raddr(ram_raddr), .rdata(ram_rdata)
  );

  // ---- ROM twiddle -------------------------------------------------------------------------------
  logic [7:0]  tw_addr;
  logic        tw_en;
  logic [TW_W-1:0] tw_cos_u, tw_sin_u;
  rom_sync #(.WIDTH(TW_W), .DEPTH(NFFT/2), .INIT_FILE(INIT_COS)) u_cos (
    .clk(clk), .en(tw_en), .addr(tw_addr), .dout(tw_cos_u));
  rom_sync #(.WIDTH(TW_W), .DEPTH(NFFT/2), .INIT_FILE(INIT_SIN)) u_sin (
    .clk(clk), .en(tw_en), .addr(tw_addr), .dout(tw_sin_u));

  // ---- fungsi bit-reversal 9 bit -----------------------------------------------------------------
  function automatic logic [AW-1:0] bitrev(input logic [AW-1:0] v);
    for (int i = 0; i < AW; i++) bitrev[i] = v[AW-1-i];
  endfunction

  // ---- LOAD --------------------------------------------------------------------------------------
  logic [AW-1:0] ld_cnt;
  wire  [AW-1:0] ld_cur  = in_first ? '0 : ld_cnt;
  wire           ld_acc  = in_valid && (state == S_LOAD);
  assign in_ready = (state == S_LOAD);

  // ---- COMPUTE: penerbit baca ----------------------------------------------------------------------
  logic [3:0]    st;                                // stage 0..8
  logic [7:0]    j;                                 // butterfly dalam stage
  logic          ph;                                // 0: baca a, 1: baca b
  logic [3:0]    stall_cnt;
  logic [8:0]    idx_a, idx_b;

  wire [8:0] pos_mask = (9'd1 << st) - 9'd1;
  wire [8:0] j9       = {1'b0, j};
  assign idx_a = ((j9 >> st) << (st + 1)) | (j9 & pos_mask);
  assign idx_b = idx_a | (9'd1 << st);
  wire [8:0] tw_full  = (j9 & pos_mask) << (4'd8 - st);   // pos * 2^(8-st) (0..255)

  wire issue0 = (state == S_COMP) && (ph == 1'b0);
  wire issue1 = (state == S_COMP) && (ph == 1'b1);

  // ---- alur butterfly (pipeline) -------------------------------------------------------------------
  // t   : issue0, raddr=a, tw ROM dibaca
  // t+1 : rdata=a, tw valid -> a_q, tw_q ; raddr=b
  // t+2 : rdata=b            -> produk (4 pengali)
  // t+3 : produk valid       -> tr, ti
  // t+4 : tr,ti valid        -> a', b' (bulatkan, saturasi); sat_inc
  // t+5 : a' ditulis ; t+6 : b' ditulis
  logic                      v1, v2, v2q, v3, v4, v5, v6, v7;
  logic [AW-1:0]             ia1, ia2, ia2q, ia3, ia4, ia5, ia6;
  logic [AW-1:0]             ib1, ib2, ib2q, ib3, ib4, ib5, ib6, ib7;
  logic signed [W-1:0]       ar_q, ai_q, ar_d, ai_d;
  logic signed [TW_W-1:0]    c_q, s_q;
  logic signed [W-1:0]       br_q, bi_q;          // b didaftarkan sebelum pengali (memutus BRAM -> DSP)
  logic signed [PW-1:0]      p_brc, p_bis, p_brs, p_bic;
  logic signed [AW_SUM-1:0]  tr_q, ti_q;
  logic signed [W-1:0]       oar, oai, obr, obi;
  logic signed [W-1:0]       obr2, obi2;
  logic signed [AW_SUM-1:0]  ar_base, ai_base;      // a*2^15 + rnd (dihitung satu tahap lebih awal, memperpendek jalur kritis)
  logic                      ss_q;                  // bit geser stage untuk tahap t+4
  logic [2:0]                sat_inc;               // jumlah saturasi butterfly ini (0..4), ditambahkan ke sat_cnt di t+5
  logic [31:0]               sat_cnt;

  wire signed [W-1:0] b_re = ram_rdata[W-1:0];
  wire signed [W-1:0] b_im = ram_rdata[2*W-1:W];
  wire signed [W-1:0] a_re = ram_rdata[W-1:0];
  wire signed [W-1:0] a_im = ram_rdata[2*W-1:W];

  // pembulatan/saturasi tahap t+4. sh hanya 15 atau 16: pemilih 2:1, bukan penggeser variabel.
  // a*2^15 + rnd sudah dihitung di t+3 (ar_base/ai_base); di sini tinggal satu penjumlah per keluaran.
  wire         ss_bit  = STAGE_MASK[st];
  wire signed [AW_SUM-1:0] rnd  = ss_bit ? (AW_SUM'(1) <<< 15) : (AW_SUM'(1) <<< 14);
  wire signed [AW_SUM-1:0] ar15 = (AW_SUM'(ar_d) <<< 15);
  wire signed [AW_SUM-1:0] ai15 = (AW_SUM'(ai_d) <<< 15);
  wire signed [AW_SUM-1:0] sum_ar_c = ar_base + tr_q;
  wire signed [AW_SUM-1:0] sum_br_c = ar_base - tr_q;
  wire signed [AW_SUM-1:0] sum_ai_c = ai_base + ti_q;
  wire signed [AW_SUM-1:0] sum_bi_c = ai_base - ti_q;
  logic signed [AW_SUM-1:0] sum_ar, sum_br, sum_ai, sum_bi;   // jumlah didaftarkan (tahap v4)
  logic                     ss_q2;
  wire signed [AW_SUM-1:0] q_ar = ss_q2 ? (sum_ar >>> 16) : (sum_ar >>> 15);
  wire signed [AW_SUM-1:0] q_br = ss_q2 ? (sum_br >>> 16) : (sum_br >>> 15);
  wire signed [AW_SUM-1:0] q_ai = ss_q2 ? (sum_ai >>> 16) : (sum_ai >>> 15);
  wire signed [AW_SUM-1:0] q_bi = ss_q2 ? (sum_bi >>> 16) : (sum_bi >>> 15);
  localparam logic signed [AW_SUM-1:0] MAXW = (AW_SUM'(1) <<< (W - 1)) - 1;
  localparam logic signed [AW_SUM-1:0] MINW = -(AW_SUM'(1) <<< (W - 1));
  wire sat_ar = (q_ar > MAXW) || (q_ar < MINW);
  wire sat_br = (q_br > MAXW) || (q_br < MINW);
  wire sat_ai = (q_ai > MAXW) || (q_ai < MINW);
  wire sat_bi = (q_bi > MAXW) || (q_bi < MINW);
  function automatic logic signed [W-1:0] clampw(input logic signed [AW_SUM-1:0] v);
    clampw = (v > MAXW) ? MAXW[W-1:0] : (v < MINW) ? MINW[W-1:0] : v[W-1:0];
  endfunction

  // ---- OUTPUT --------------------------------------------------------------------------------------
  logic [8:0]  o_cnt;
  logic [3:0]  o_gap;
  logic        o_rd_v;                              // pembacaan RAM tertunda 1 siklus
  logic [8:0]  o_rd_idx;

  // ---- kontrol utama -------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state     <= S_LOAD;
      ld_cnt    <= '0;
      st        <= '0;
      j         <= '0;
      ph        <= 1'b0;
      stall_cnt <= '0;
      o_cnt     <= '0;
      o_gap     <= '0;
      done      <= 1'b0;
      sat_cnt   <= '0;
      n_sat     <= '0;
    end else begin
      done <= 1'b0;
      case (state)
        S_LOAD: begin
          if (ld_acc) begin
            ld_cnt <= ld_cur + 9'd1;
            if (ld_cur == AW'(NFFT - 1)) begin
              state   <= S_COMP;
              st      <= '0;
              j       <= '0;
              ph      <= 1'b0;
              sat_cnt <= '0;
            end
          end
        end
        S_COMP: begin
          ph <= ~ph;
          if (ph == 1'b1) begin
            if (j == 8'(NBF - 1)) begin
              j         <= '0;
              state     <= S_STALL;
              stall_cnt <= '0;
            end else begin
              j <= j + 8'd1;
            end
          end
        end
        S_STALL: begin
          if (stall_cnt == 4'(DRAIN - 1)) begin
            if (st == 4'(STAGES - 1)) begin
              state <= S_OUT;
              o_cnt <= '0;
              o_gap <= '0;
            end else begin
              st    <= st + 4'd1;
              state <= S_COMP;
              ph    <= 1'b0;
            end
          end else begin
            stall_cnt <= stall_cnt + 4'd1;
          end
        end
        S_OUT: begin
          if (o_gap == 4'(OUT_GAP - 1)) begin
            o_gap <= '0;
            if (o_cnt == 9'(NBINS - 1)) begin
              state <= S_DONE;
            end else begin
              o_cnt <= o_cnt + 9'd1;
            end
          end else begin
            o_gap <= o_gap + 4'd1;
          end
        end
        S_DONE: begin
          done   <= 1'b1;
          n_sat  <= sat_cnt;
          state  <= S_LOAD;
          ld_cnt <= '0;
        end
        default: state <= S_LOAD;
      endcase

      // saturasi: dihitung di t+4 (sat_inc), ditambahkan ke penghitung di t+5 (v5)
      if (v6) sat_cnt <= sat_cnt + 32'(sat_inc);
    end
  end

  // ---- RAM write mux: LOAD / butterfly / (tidak ada tulis saat OUTPUT) -------------------------------
  wire wr_a = v6;                                   // tulis a'
  wire wr_b = v7;                                   // tulis b'
  always_comb begin
    ram_we    = 1'b0;
    ram_waddr = '0;
    ram_wdata = '0;
    if (ld_acc) begin
      ram_we    = 1'b1;
      ram_waddr = bitrev(ld_cur);
      ram_wdata = {{W{1'b0}}, in_data};
    end else if (wr_a) begin
      ram_we    = 1'b1;
      ram_waddr = ia6;
      ram_wdata = {oai, oar};
    end else if (wr_b) begin
      ram_we    = 1'b1;
      ram_waddr = ib7;
      ram_wdata = {obi2, obr2};
    end
  end

  // ---- RAM read mux ----------------------------------------------------------------------------------
  always_comb begin
    ram_re    = 1'b0;
    ram_raddr = '0;
    if (issue0) begin
      ram_re    = 1'b1;
      ram_raddr = idx_a;
    end else if (issue1) begin
      ram_re    = 1'b1;
      ram_raddr = idx_b;
    end else if ((state == S_OUT) && (o_gap == 4'd0)) begin
      ram_re    = 1'b1;
      ram_raddr = o_cnt;
    end
  end
  assign tw_en   = issue0;
  assign tw_addr = tw_full[7:0];

  // ---- pipeline butterfly -----------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      v1 <= 1'b0; v2 <= 1'b0; v2q <= 1'b0; v3 <= 1'b0; v4 <= 1'b0; v5 <= 1'b0; v6 <= 1'b0; v7 <= 1'b0;
    end else begin
      v1 <= issue0;
      v2 <= v1;
      v2q <= v2;
      v3 <= v2q;
      v4 <= v3;
      v5 <= v4;
      v6 <= v5;
      v7 <= v6;
    end
  end

  always_ff @(posedge clk) begin
    // alamat dibawa sepanjang alur
    ia1 <= idx_a;  ib1 <= idx_b;
    ia2 <= ia1;    ib2 <= ib1;
    ia2q <= ia2;   ib2q <= ib2;
    ia3 <= ia2q;   ib3 <= ib2q;
    ia4 <= ia3;    ib4 <= ib3;
    ia5 <= ia4;    ib5 <= ib4;
    ia6 <= ia5;    ib6 <= ib5;
    ib7 <= ib6;
    // t+1: a dan twiddle ditangkap
    if (v1) begin
      ar_q <= a_re;
      ai_q <= a_im;
      c_q  <= tw_cos_u;
      s_q  <= tw_sin_u;
    end
    // t+2: b tersedia di rdata -> didaftarkan
    if (v2) begin
      br_q <= b_re;
      bi_q <= b_im;
      ar_d <= ar_q;
      ai_d <= ai_q;
    end
    // t+3: produk (br_q/bi_q dan c_q/s_q stabil; twiddle butterfly berikutnya baru ditangkap 1 siklus kemudian)
    if (v2q) begin
      p_brc <= br_q * c_q;
      p_bis <= bi_q * s_q;
      p_brs <= br_q * s_q;
      p_bic <= bi_q * c_q;
    end
    // t+4: tr = br*cos + bi*sin ; ti = bi*cos - br*sin
    if (v3) begin
      tr_q    <= AW_SUM'(p_brc) + AW_SUM'(p_bis);
      ti_q    <= AW_SUM'(p_bic) - AW_SUM'(p_brs);
      ar_base <= ar15 + rnd;
      ai_base <= ai15 + rnd;
      ss_q    <= ss_bit;
    end
    // t+5: jumlah a*2^15 + rnd +/- t didaftarkan
    if (v4) begin
      sum_ar <= sum_ar_c;
      sum_br <= sum_br_c;
      sum_ai <= sum_ai_c;
      sum_bi <= sum_bi_c;
      ss_q2  <= ss_q;
    end
    // t+6: geser (pemilih 2:1), saturasi, hitung saturasi
    if (v5) begin
      oar <= clampw(q_ar);
      oai <= clampw(q_ai);
      obr <= clampw(q_br);
      obi <= clampw(q_bi);
      sat_inc <= 3'(sat_ar) + 3'(sat_br) + 3'(sat_ai) + 3'(sat_bi);
    end
    // t+7: simpan b' untuk ditulis di t+8
    if (v6) begin
      obr2 <= obr;
      obi2 <= obi;
    end
  end

  // ---- keluaran bin ---------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      out_valid <= 1'b0;
      out_last  <= 1'b0;
      o_rd_v    <= 1'b0;
    end else begin
      o_rd_v    <= (state == S_OUT) && (o_gap == 4'd0);
      o_rd_idx  <= o_cnt;
      out_valid <= o_rd_v;
      out_last  <= o_rd_v && (o_rd_idx == 9'(NBINS - 1));
      if (o_rd_v) begin
        out_re  <= ram_rdata[W-1:0];
        out_im  <= ram_rdata[2*W-1:W];
        out_idx <= o_rd_idx;
      end
    end
  end

`ifndef SYNTHESIS
  initial begin
    if (DRAIN < 10)  $error("fft512: DRAIN=%0d < 10", DRAIN);
    if (OUT_GAP < 3) $error("fft512: OUT_GAP=%0d < 3", OUT_GAP);
  end
  always_ff @(posedge clk) begin
    if (rst_n && in_valid && state != S_LOAD)
      $error("fft512: sampel masukan datang saat tidak dalam fase LOAD");
  end
`endif
endmodule
