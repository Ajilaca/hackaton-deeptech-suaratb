// cnn_top: akselerator CNN int8 (conv3x3 x3 + ReLU + MaxPool, GAP, FC 32->16->1), output-stationary, P lane.
// Fungsi    : memuat 40x50 int8 ke buffer A (ld_*), lalu `start`; mengeluarkan logit int16 (done 1 siklus).
//             Buffer A/B (ping-pong) berupa RAM inferensi; bobot/bias/requant dari ROM (.mem dikemas P lane).
// Parameter : INIT_W, INIT_B, INIT_RQ (berkas .mem hasil golden/gen_cnn_rom.py), bentuk layer dari cnn_cfg_pkg
// Antarmuka : ld_valid/ld_addr/ld_q (urutan CNN: addr = mel*50 + frame), start, busy, done, logit
// Latensi   : ~230 rb siklus pada P=16 (target/estimasi; ukur di Fase T/6)
// Golden    : golden/blocks/cnn.py : cnn_forward(x, model)
module cnn_top #(
  parameter INIT_W  = "tb/vectors/cnn/cnn_wrom.mem",
  parameter INIT_B  = "tb/vectors/cnn/cnn_bias.mem",
  parameter INIT_RQ = "tb/vectors/cnn/cnn_rq.mem"
) (
  input  logic                clk,
  input  logic                rst_n,
  input  logic                ld_valid,
  input  logic [11:0]         ld_addr,
  input  logic signed [7:0]   ld_q,
  input  logic                start,
  output logic                busy,
  output logic                done,
  output logic signed [15:0]  logit
);
  import cnn_cfg_pkg::*;

  localparam int AW   = $clog2(B_DEPTH);          // 13
  localparam int AWA  = $clog2(A_DEPTH);          // 12
  localparam int ACCW = suaratb_pkg::ACC_BITS;
  localparam int BW   = suaratb_pkg::BIAS_BITS;

  // ---- kontrol ----------------------------------------------------------------------------------------
  logic                  c_v1, c_inb1, c_first1, c_lastpos1, c_posfirst1, c_pixlast1;
  logic                  c_rd_en, c_rd_src_b, c_wrom_en, c_brom_en;
  logic [AW-1:0]         c_rd_addr, cap_base, cap_stride;
  logic [$clog2(WROM_DEPTH)-1:0] c_wrom_addr;
  logic [$clog2(NBG)-1:0]        c_brom_addr;
  logic [7:0]            cap_lanes;
  logic [2:0]            layer;
  logic                  wr_done, gap_start, gap_done, in_gap;

  cnn_ctrl #(.AW(AW)) u_ctrl (
    .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done),
    .v1(c_v1), .inb1(c_inb1), .first1(c_first1), .lastpos1(c_lastpos1), .posfirst1(c_posfirst1), .pixlast1(c_pixlast1),
    .rd_en(c_rd_en), .rd_addr(c_rd_addr), .rd_src_b(c_rd_src_b),
    .wrom_en(c_wrom_en), .wrom_addr(c_wrom_addr), .brom_en(c_brom_en), .brom_addr(c_brom_addr),
    .cap_base(cap_base), .cap_lanes(cap_lanes), .cap_stride(cap_stride), .layer(layer),
    .wr_done(wr_done), .gap_start(gap_start), .gap_done(gap_done), .in_gap(in_gap)
  );

  // ---- ROM -------------------------------------------------------------------------------------------
  logic [P*8-1:0]    w_word;
  logic [P*BW-1:0]   b_word;
  rom_sync #(.WIDTH(P*8), .DEPTH(WROM_DEPTH), .INIT_FILE(INIT_W)) u_wrom (
    .clk(clk), .en(c_wrom_en), .addr(c_wrom_addr), .dout(w_word));
  rom_sync #(.WIDTH(P*BW), .DEPTH(NBG), .INIT_FILE(INIT_B)) u_brom (
    .clk(clk), .en(c_brom_en), .addr(c_brom_addr), .dout(b_word));

  logic [15:0] rq_mem [0:2*NL-1];
  initial $readmemh(INIT_RQ, rq_mem);
  // konfigurasi layer didaftarkan (tetap selama satu layer; layer berubah >= 1 siklus sebelum token pertama)
  logic [15:0] rq_m0;
  logic [4:0]  rq_shift;
  logic        rq_final, rq_dst_b;
  always_ff @(posedge clk) begin
    rq_m0    <= rq_mem[{layer, 1'b0}];
    rq_shift <= rq_mem[{layer, 1'b1}][4:0];
    rq_final <= (cfg_final(int'(layer)) != 0);
    rq_dst_b <= (cfg_dst_b(int'(layer)) != 0);
  end

  // ---- buffer A/B ---------------------------------------------------------------------------------------
  logic              gap_rd_en, gap_we;
  logic [AW-1:0]     gap_rd_addr, gap_waddr;
  logic signed [7:0] gap_wdata;
  logic              rq_we;
  logic [AW-1:0]     rq_waddr;
  logic signed [15:0] rq_wdata;

  logic signed [7:0] rdA, rdB;
  wire  a_we    = (ld_valid && !busy) || gap_we || (rq_we && !rq_dst_b && !rq_final);
  wire  [AWA-1:0] a_waddr = (ld_valid && !busy) ? ld_addr[AWA-1:0] : gap_we ? gap_waddr[AWA-1:0] : rq_waddr[AWA-1:0];
  wire  signed [7:0] a_wdata = (ld_valid && !busy) ? ld_q : gap_we ? gap_wdata : rq_wdata[7:0];
  wire  b_we    = rq_we && rq_dst_b && !rq_final;
  wire  a_re    = c_rd_en && !c_rd_src_b;
  wire  b_re    = (c_rd_en && c_rd_src_b) || gap_rd_en;
  wire  [AW-1:0] b_raddr = gap_rd_en ? gap_rd_addr : c_rd_addr;

  ram_sdp #(.WIDTH(8), .DEPTH(A_DEPTH)) u_ramA (
    .clk(clk), .we(a_we), .waddr(a_waddr), .wdata(a_wdata), .re(a_re), .raddr(c_rd_addr[AWA-1:0]), .rdata(rdA));
  ram_sdp #(.WIDTH(8), .DEPTH(B_DEPTH)) u_ramB (
    .clk(clk), .we(b_we), .waddr(rq_waddr), .wdata(rq_wdata[7:0]), .re(b_re), .raddr(b_raddr), .rdata(rdB));

  // ---- GAP ----------------------------------------------------------------------------------------------
  gap #(.AW(AW)) u_gap (
    .clk(clk), .rst_n(rst_n), .start(gap_start), .rd_en(gap_rd_en), .rd_addr(gap_rd_addr), .rd_data(rdB),
    .we(gap_we), .waddr(gap_waddr), .wdata(gap_wdata), .done(gap_done));

  // ---- token tahap 2 ------------------------------------------------------------------------------------
  logic v2, inb2, first2, lastpos2, posfirst2, pixlast2, src_b2;
  always_ff @(posedge clk) begin
    if (!rst_n) v2 <= 1'b0;
    else        v2 <= c_v1;
    inb2      <= c_inb1;
    first2    <= c_first1;
    lastpos2  <= c_lastpos1;
    posfirst2 <= c_posfirst1;
    pixlast2  <= c_pixlast1;
    src_b2    <= c_rd_src_b;
  end
  wire signed [7:0] act_sel = src_b2 ? rdB : rdA;
  wire signed [7:0] act     = inb2 ? act_sel : 8'sd0;

  // ---- PE array -------------------------------------------------------------------------------------------
  logic [P*ACCW-1:0] res;
  logic              res_valid;
  pe_array #(.P(P)) u_pe (
    .clk(clk), .rst_n(rst_n), .in_valid(v2), .in_first(first2), .in_lastpos(lastpos2), .in_posfirst(posfirst2),
    .in_pixlast(pixlast2), .in_act(act), .in_w(w_word), .in_bias(b_word), .res(res), .res_valid(res_valid));

  // ---- requant + tulis ---------------------------------------------------------------------------------
  requant #(.P(P), .AW(AW)) u_rq (
    .clk(clk), .rst_n(rst_n), .start(res_valid), .res(res), .lanes(cap_lanes), .base(cap_base), .stride(cap_stride),
    .m0(rq_m0), .shift(rq_shift), .final_l(rq_final),
    .we(rq_we), .waddr(rq_waddr), .wdata(rq_wdata), .done(wr_done));

  always_ff @(posedge clk) begin
    if (!rst_n) logit <= '0;
    else if (rq_we && rq_final) logit <= rq_wdata;
  end

`ifndef SYNTHESIS
  always_ff @(posedge clk) begin
    if (rst_n && res_valid && u_rq.run) $error("cnn_top: hasil pixel baru tiba saat requant belum selesai");
  end
`endif
endmodule
