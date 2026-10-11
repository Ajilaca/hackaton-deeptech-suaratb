// cnn_ctrl: sekuenser layer CNN output-stationary: penerbit token MAC (alamat aktivasi, bobot, bias) dan urutan layer.
// Fungsi    : urutan layer = conv1, conv2, conv3, GAP, fc1, fc2 (konfigurasi dari cnn_cfg_pkg). Untuk tiap layer mesin
//             loop (dalam -> luar): tx, ty, ci, pw (posisi jendela pool 0..3 atau 0), ox, oy, g (grup P kanal).
//             Tiap token = satu MAC per lane. Padding 'same': alamat di luar batas -> inb=0 (aktivasi nol).
//             Setelah token pixlast (posisi terakhir pool) hasil dikirim ke requant; capture alamat/lanes ada di cap_*.
// Parameter : dari cnn_cfg_pkg
// Antarmuka : start -> busy, done; token tahap-1: v1/inb1/first1/lastpos1/posfirst1/pixlast1 + rd_en/rd_addr/rd_src_b,
//             wrom_en/wrom_addr, brom_en/brom_addr; cap_base/cap_lanes; layer; wr_done (dari requant), gap_start/gap_done.
// Latensi   : token 1/siklus; ~230 rb siklus total pada P=16 (analitik: 18.000 + 138.240 + 69.120 + ~1,5 rb + overhead tulis)
// Golden    : golden/blocks/cnn.py : cnn_forward
module cnn_ctrl #(
  parameter int AW = 13
) (
  input  logic                clk,
  input  logic                rst_n,
  input  logic                start,
  output logic                busy,
  output logic                done,
  // token tahap 1
  output logic                v1,
  output logic                inb1,
  output logic                first1,
  output logic                lastpos1,
  output logic                posfirst1,
  output logic                pixlast1,
  output logic                rd_en,
  output logic [AW-1:0]       rd_addr,
  output logic                rd_src_b,
  output logic                wrom_en,
  output logic [$clog2(cnn_cfg_pkg::WROM_DEPTH)-1:0] wrom_addr,
  output logic                brom_en,
  output logic [$clog2(cnn_cfg_pkg::NBG)-1:0]        brom_addr,
  // penulisan
  output logic [AW-1:0]       cap_base,
  output logic [7:0]          cap_lanes,
  output logic [AW-1:0]       cap_stride,
  output logic [2:0]          layer,
  input  logic                wr_done,
  // GAP
  output logic                gap_start,
  input  logic                gap_done,
  output logic                in_gap
);
  import cnn_cfg_pkg::*;

  localparam int WA = $clog2(WROM_DEPTH);
  localparam int BA = $clog2(NBG);

  typedef enum logic [2:0] {S_IDLE, S_INIT, S_ENG, S_DRAIN, S_GAP, S_FIN} st_t;
  st_t state;

  // konfigurasi layer aktif
  logic [7:0] c_cin, c_cout, c_h, c_w, c_ho, c_wo, c_g, c_T;
  logic [1:0] c_kmax;
  logic       c_pool, c_src_b, c_dst_b;
  logic [WA-1:0] c_wbase;
  logic [BA-1:0] c_bbase;
  always_comb begin
    c_cin  = 8'(cfg_cin(int'(layer)));
    c_cout = 8'(cfg_cout(int'(layer)));
    c_h    = 8'(cfg_h(int'(layer)));
    c_w    = 8'(cfg_w(int'(layer)));
    c_ho   = 8'(cfg_ho(int'(layer)));
    c_wo   = 8'(cfg_wo(int'(layer)));
    c_g    = 8'(cfg_G(int'(layer)));
    c_T    = 8'(cfg_T(int'(layer)));
    c_kmax = 2'(cfg_k(int'(layer)) - 1);
    c_pool = cfg_pool(int'(layer)) != 0;
    c_src_b = cfg_src_b(int'(layer)) != 0;
    c_dst_b = cfg_dst_b(int'(layer)) != 0;
    c_wbase = WA'(cfg_wbase(int'(layer)));
    c_bbase = BA'(cfg_bbase(int'(layer)));
  end
  wire [7:0]  c_npos  = c_pool ? 8'd4 : 8'd1;
  wire [AW-1:0] c_hw  = AW'(c_h) * AW'(c_w);
  wire [AW-1:0] c_howo = AW'(c_ho) * AW'(c_wo);
  wire        pad_on  = (c_kmax != 2'd0);

  // penghitung (tahap 0)
  logic [1:0]    tx, ty, pw;
  logic [7:0]    ci, ox, oy, gg;
  logic [AW-1:0] cibase;
  logic [WA-1:0] wptr, wp_pos0;
  logic [AW-1:0] gout_base, pix;
  logic [7:0]    lanes_cur;

  wire tx_last = (tx == c_kmax);
  wire ty_last = (ty == c_kmax);
  wire ci_last = (ci == c_cin - 8'd1);
  wire pw_last = (pw == c_npos[1:0] - 2'd1);
  wire ox_last = (ox == c_wo - 8'd1);
  wire oy_last = (oy == c_ho - 8'd1);
  wire g_last  = (gg == c_g - 8'd1);
  wire tok_lastpos   = tx_last && ty_last && ci_last;
  wire tok_pixlast   = tok_lastpos && pw_last;
  wire tok_layerlast = tok_pixlast && ox_last && oy_last && g_last;

  // koordinat masukan (tahap 0)
  wire [7:0] dy = c_pool ? {7'd0, pw[1]} : 8'd0;
  wire [7:0] dx = c_pool ? {7'd0, pw[0]} : 8'd0;
  wire [7:0] oyc = c_pool ? {oy[6:0], 1'b0} + dy : oy;
  wire [7:0] oxc = c_pool ? {ox[6:0], 1'b0} + dx : ox;
  wire signed [9:0] iy0 = $signed({2'b00, oyc}) + $signed({8'd0, ty}) - (pad_on ? 10'sd1 : 10'sd0);
  wire signed [9:0] ix0 = $signed({2'b00, oxc}) + $signed({8'd0, tx}) - (pad_on ? 10'sd1 : 10'sd0);
  wire inb0 = (iy0 >= 0) && (iy0 < $signed({2'b00, c_h})) && (ix0 >= 0) && (ix0 < $signed({2'b00, c_w}));

  // tahap 1 (register)
  logic [7:0]    iy1, ix1;
  logic [AW-1:0] cibase1;
  logic [WA-1:0] wptr1;
  logic [BA-1:0] bidx1;
  logic          src_b1;
  logic          s1_v, s1_inb, s1_first, s1_lastpos, s1_posfirst, s1_pixlast;
  logic          run_tok;
  assign run_tok = (state == S_ENG);

  // tahap 2: offset baris iy*W; tahap 3: alamat penuh didaftarkan (BRAM/ROM mendapat alamat dan enable dari register)
  logic          s2_v, s2_inb, s2_first, s2_lastpos, s2_posfirst, s2_pixlast, s2_src_b;
  logic [AW-1:0] s2_rowoff, s2_cibase, s2_ix;
  logic [WA-1:0] s2_wptr;
  logic [BA-1:0] s2_bidx;
  logic [AW-1:0] c_hw_q, c_howo_q, c_w_q;

  assign in_gap     = (state == S_GAP);
  assign busy       = (state != S_IDLE) && (state != S_FIN);
  assign cap_stride = c_howo_q;

  always_ff @(posedge clk) begin
    c_hw_q   <= c_hw;
    c_howo_q <= c_howo;
    c_w_q    <= AW'(c_w);
    // tahap 2
    s2_v        <= s1_v && rst_n;
    s2_inb      <= s1_inb;
    s2_first    <= s1_first;
    s2_lastpos  <= s1_lastpos;
    s2_posfirst <= s1_posfirst;
    s2_pixlast  <= s1_pixlast;
    s2_src_b    <= src_b1;
    s2_rowoff   <= AW'(iy1) * c_w_q;
    s2_cibase   <= cibase1;
    s2_ix       <= AW'(ix1);
    s2_wptr     <= wptr1;
    s2_bidx     <= bidx1;
    // tahap 3 (keluaran modul = "tahap 1" bagi cnn_top)
    v1        <= s2_v && rst_n;
    inb1      <= s2_inb;
    first1    <= s2_first;
    lastpos1  <= s2_lastpos;
    posfirst1 <= s2_posfirst;
    pixlast1  <= s2_pixlast;
    rd_src_b  <= s2_src_b;
    rd_en     <= rst_n && s2_v && s2_inb && (state != S_GAP);
    rd_addr   <= s2_cibase + s2_rowoff + s2_ix;
    wrom_en   <= rst_n && s2_v;
    wrom_addr <= s2_wptr;
    brom_en   <= rst_n && s2_v && s2_first;
    brom_addr <= s2_bidx;
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= S_IDLE;
      layer <= '0;
      s1_v <= 1'b0;
      done <= 1'b0;
      gap_start <= 1'b0;
      tx <= '0; ty <= '0; pw <= '0; ci <= '0; ox <= '0; oy <= '0; gg <= '0;
      cibase <= '0; wptr <= '0; wp_pos0 <= '0; gout_base <= '0; pix <= '0; lanes_cur <= '0;
    end else begin
      done      <= 1'b0;
      gap_start <= 1'b0;
      s1_v        <= 1'b0;

      case (state)
        S_IDLE: begin
          if (start) begin
            layer <= '0;
            state <= S_INIT;
          end
        end
        S_INIT: begin
          tx <= '0; ty <= '0; pw <= '0; ci <= '0; ox <= '0; oy <= '0; gg <= '0;
          cibase    <= '0;
          wptr      <= c_wbase;
          wp_pos0   <= c_wbase;
          gout_base <= '0;
          pix       <= '0;
          lanes_cur <= (c_cout > 8'(cnn_cfg_pkg::P)) ? 8'(cnn_cfg_pkg::P) : c_cout;
          state     <= S_ENG;
        end
        S_ENG: begin
          // keluarkan token tahap 1
          s1_v        <= 1'b1;
          iy1       <= iy0[7:0];
          ix1       <= ix0[7:0];
          s1_inb      <= inb0;
          cibase1   <= cibase;
          wptr1     <= wptr;
          bidx1     <= c_bbase + BA'(gg);
          src_b1    <= c_src_b;
          s1_first    <= (tx == 2'd0) && (ty == 2'd0) && (ci == 8'd0);
          s1_lastpos  <= tok_lastpos;
          s1_posfirst <= (pw == 2'd0);
          s1_pixlast  <= tok_pixlast;
          if (tok_pixlast) begin
            cap_base  <= gout_base + pix;
            cap_lanes <= lanes_cur;
          end
          // majukan penghitung
          wptr <= wptr + 1'b1;
          if (!tx_last) begin
            tx <= tx + 2'd1;
          end else begin
            tx <= '0;
            if (!ty_last) begin
              ty <= ty + 2'd1;
            end else begin
              ty <= '0;
              if (!ci_last) begin
                ci     <= ci + 8'd1;
                cibase <= cibase + c_hw_q;
              end else begin
                ci     <= '0;
                cibase <= '0;
                wptr   <= wp_pos0;                          // posisi berikutnya memakai deret bobot yang sama
                if (!pw_last) begin
                  pw <= pw + 2'd1;
                end else begin
                  pw  <= '0;
                  pix <= pix + 1'b1;
                  if (!ox_last) begin
                    ox <= ox + 8'd1;
                  end else begin
                    ox <= '0;
                    if (!oy_last) begin
                      oy <= oy + 8'd1;
                    end else begin
                      oy        <= '0;
                      pix       <= '0;
                      gg        <= gg + 8'd1;
                      gout_base <= gout_base + AW'(cnn_cfg_pkg::P) * c_howo_q;
                      wp_pos0   <= wp_pos0 + (WA'(c_cin) * WA'(c_T));
                      wptr      <= wp_pos0 + (WA'(c_cin) * WA'(c_T));
                      lanes_cur <= ((gg + 8'd1) == c_g - 8'd1) ? (c_cout - (gg + 8'd1) * 8'(cnn_cfg_pkg::P)) : 8'(cnn_cfg_pkg::P);
                    end
                  end
                end
              end
            end
          end
          if (tok_layerlast) state <= S_DRAIN;
        end
        S_DRAIN: begin
          if (wr_done) begin
            if (layer == 3'd2) begin
              state     <= S_GAP;
              gap_start <= 1'b1;
            end else if (layer == 3'd4) begin
              state <= S_FIN;
              done  <= 1'b1;
            end else begin
              layer <= layer + 3'd1;
              state <= S_INIT;
            end
          end
        end
        S_GAP: begin
          if (gap_done) begin
            layer <= 3'd3;
            state <= S_INIT;
          end
        end
        S_FIN: begin
          state <= S_IDLE;
        end
        default: state <= S_IDLE;
      endcase
    end
  end
endmodule
