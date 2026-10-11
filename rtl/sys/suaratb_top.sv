// suaratb_top: sistem lengkap audio -> skor risiko. Sumber sampel: mikrofon I2S (mode 1) atau UART dataset (mode 0).
// Alur     : i2s_rx/ds -> preemph -> frame_buf -> dsp_chain -> feat_buf -> cnn_top -> logit (UART 'L' + CSR)
//            frame_buf.e_out -> cough_det (mode mikrofon: trig -> segmen start..end -> inferensi)
//            mode dataset: setelah 50 frame fitur -> segmen mulai frame 0 -> inferensi
// Parameter : CLK_PER_BIT (UART), CLK_PER_BCLK (I2S; 96 pada 98,304 MHz), INIT_* (.mem CNN)
// Antarmuka : clk (inti), rst_n, i2s (bclk/ws/sd), uart (rx/tx), led_busy/led_rec
// Catatan   : tanpa MMCM/pin; target/ menambahkan clock, pin, dan constraint.
module suaratb_top #(
  parameter int CLK_PER_BIT  = 853,
  parameter int CLK_PER_BCLK = suaratb_pkg::FS == 16000 ? 96 : 96,
  parameter     INIT_W  = "tb/vectors/cnn/cnn_wrom.mem",
  parameter     INIT_B  = "tb/vectors/cnn/cnn_bias.mem",
  parameter     INIT_RQ = "tb/vectors/cnn/cnn_rq.mem",
  parameter     INIT_WIN = "rtl/mem/window_hamming.mem",
  parameter     INIT_COS = "rtl/mem/twiddle_cos.mem",
  parameter     INIT_SIN = "rtl/mem/twiddle_sin.mem",
  parameter     INIT_MEL = "rtl/mem/mel_rom.mem",
  parameter     INIT_LOG = "rtl/mem/log_lut.mem"
) (
  input  logic clk,
  input  logic rst_n,
  output logic bclk,
  output logic ws,
  input  logic sd,
  input  logic uart_rx,
  output logic uart_tx,
  output logic led_busy,
  output logic led_rec
);
  localparam int PF = suaratb_pkg::PREEMPH_FRAC;
  localparam int W  = suaratb_pkg::FFT_W;

  // ---- CSR dan UART --------------------------------------------------------------------------------
  logic        csr_we, csr_rd;
  logic [7:0]  csr_waddr, csr_wdata, csr_raddr, csr_rdata, seq;
  logic        ds_start, ds_valid;
  logic signed [15:0] ds_sample;
  logic        res_valid;
  logic signed [15:0] res_logit;
  logic        mode_mic, clear_pulse;
  logic        cnn_busy, recording;
  logic [15:0] frame_count16, nsat16;
  logic        nsat_valid;

  uart_bridge #(.CLK_PER_BIT(CLK_PER_BIT)) u_br (
    .clk(clk), .rst_n(rst_n), .rx(uart_rx), .tx(uart_tx),
    .csr_we(csr_we), .csr_waddr(csr_waddr), .csr_wdata(csr_wdata), .csr_raddr(csr_raddr), .csr_rdata(csr_rdata), .csr_rd(csr_rd),
    .ds_start(ds_start), .ds_valid(ds_valid), .ds_sample(ds_sample),
    .res_valid(res_valid), .res_logit(res_logit), .res_seq(seq));

  csr u_csr (
    .clk(clk), .rst_n(rst_n), .we(csr_we), .waddr(csr_waddr), .wdata(csr_wdata), .raddr(csr_raddr), .rdata(csr_rdata),
    .rd_strobe(csr_rd), .cnn_busy(cnn_busy), .recording(recording), .res_valid(res_valid), .res_logit(res_logit),
    .nsat_frame(nsat16), .nsat_valid(nsat_valid), .frame_count(frame_count16),
    .mode_mic(mode_mic), .clear_pulse(clear_pulse), .seq(seq));

  // ---- reset lunak untuk jalur DSP (ds_start / clear) ---------------------------------------------------
  logic [2:0] clr_cnt;
  logic       mode_q;                                     // pindah mode membersihkan jalur DSP (sisa sampel mode lama)
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      clr_cnt <= '0;
      mode_q  <= 1'b0;
    end else begin
      mode_q <= mode_mic;
      if (ds_start || clear_pulse || (mode_mic != mode_q)) clr_cnt <= 3'd5;
      else if (clr_cnt != 0) clr_cnt <= clr_cnt - 3'd1;
    end
  end
  wire core_rst_n = rst_n && (clr_cnt == 0);


  // ---- sumber sampel -------------------------------------------------------------------------------
  logic        mic_valid;
  logic signed [15:0] mic_sample;
  i2s_rx #(.CLK_PER_BCLK(CLK_PER_BCLK)) u_i2s (
    .clk(clk), .rst_n(rst_n), .en(mode_mic), .bclk(bclk), .ws(ws), .sd(sd),
    .sample(mic_sample), .sample_valid(mic_valid));

  wire        s_valid  = mode_mic ? mic_valid  : ds_valid;
  wire signed [15:0] s_sample = mode_mic ? mic_sample : ds_sample;

  // ---- front-end -------------------------------------------------------------------------------------
  logic        pe_valid;
  logic signed [15+PF:0] pe_wide;
  logic signed [15:0]    pe_s16;
  preemph u_pe (.clk(clk), .rst_n(core_rst_n), .in_valid(s_valid), .in_sample(s_sample),
                .out_valid(pe_valid), .out_wide(pe_wide), .out_s16(pe_s16));

  logic        fr_valid, fr_first, fr_last, e_valid;
  logic signed [15+PF:0] fr_data;
  logic [31:0] fr_idx;
  logic [47:0] e_out;
  frame_buf u_fb (.clk(clk), .rst_n(core_rst_n), .in_valid(pe_valid), .in_wide(pe_wide), .in_s16(pe_s16),
                  .frame_valid(fr_valid), .frame_data(fr_data), .frame_first(fr_first), .frame_last(fr_last),
                  .frame_idx(fr_idx), .e_valid(e_valid), .e_out(e_out));

  logic        det_trig, det_segdone;
  logic [31:0] det_onset, det_start, det_end;
  cough_det u_det (.clk(clk), .rst_n(core_rst_n), .e_valid(e_valid), .e(e_out), .trig(det_trig), .onset_frame(det_onset),
                   .start_frame(det_start), .end_frame(det_end), .seg_done(det_segdone), .recording(recording));

  // ---- DSP + feat_buf ---------------------------------------------------------------------------------
  logic        q_valid, q_last, fft_done;
  logic signed [7:0] q_data;
  logic [5:0]  q_idx;
  logic [31:0] n_sat;
  dsp_chain #(.INIT_WIN(INIT_WIN), .INIT_COS(INIT_COS), .INIT_SIN(INIT_SIN), .INIT_MEL(INIT_MEL), .INIT_LOG(INIT_LOG)) u_dsp (
    .clk(clk), .rst_n(core_rst_n), .in_valid(fr_valid), .in_data(fr_data), .in_first(fr_first),
    .out_valid(q_valid), .out_q(q_data), .out_idx(q_idx), .out_last(q_last), .fft_done(fft_done), .n_sat(n_sat));

  logic        seg_go, fb_valid, fb_last, seg_busy;
  logic signed [7:0] fb_q;
  logic [10:0] fb_addr;
  logic [31:0] seg_start, frame_count;
  feat_buf u_feat (.clk(clk), .rst_n(core_rst_n), .in_valid(q_valid), .in_q(q_data), .in_idx(q_idx), .in_last(q_last),
                   .seg_go(seg_go), .seg_start(seg_start), .out_valid(fb_valid), .out_q(fb_q), .out_addr(fb_addr),
                   .out_last(fb_last), .seg_busy(seg_busy), .frame_count(frame_count));
  assign frame_count16 = frame_count[15:0];
  assign nsat16        = n_sat[15:0];
  assign nsat_valid    = fft_done;

  // ---- sekuenser segmen -> inferensi -----------------------------------------------------------------
  typedef enum logic [1:0] {P_IDLE, P_WAIT, P_READ, P_INFER} p_t;
  p_t          p_state;
  logic [31:0] p_start, p_end;
  logic        cnn_start, cnn_done;
  logic signed [15:0] logit;
  logic        ds_arm;                                    // klip dataset baru menunggu diproses
  always_ff @(posedge clk) begin
    if (!rst_n) ds_arm <= 1'b0;
    else if (ds_start) ds_arm <= 1'b1;
    else if (core_rst_n && p_state == P_IDLE && !mode_mic && ds_arm && frame_count >= 32'(suaratb_pkg::N_FRAMES)) ds_arm <= 1'b0;
  end

  always_ff @(posedge clk) begin
    seg_go    <= 1'b0;
    cnn_start <= 1'b0;
    res_valid <= 1'b0;
    if (!rst_n || !core_rst_n) begin
      p_state <= P_IDLE;
    end else begin
      case (p_state)
        P_IDLE: begin
          if (mode_mic && det_trig) begin
            p_start <= det_start;
            p_end   <= det_start + 32'(suaratb_pkg::N_FRAMES - 1);   // start terpotong di 0: tunggu sampai frame ke-N
            p_state <= P_WAIT;
          end else if (!mode_mic && ds_arm && frame_count >= 32'(suaratb_pkg::N_FRAMES)) begin
            p_start <= 32'd0;
            p_end   <= 32'(suaratb_pkg::N_FRAMES - 1);
            p_state <= P_WAIT;
          end
        end
        P_WAIT: begin                                 // tunggu fitur frame terakhir segmen tertulis
          if (frame_count > p_end && !cnn_busy) begin
            seg_go    <= 1'b1;
            seg_start <= p_start;
            p_state   <= P_READ;
          end
        end
        P_READ: begin
          if (fb_valid && fb_last) begin
            cnn_start <= 1'b1;
            p_state   <= P_INFER;
          end
        end
        P_INFER: begin
          if (cnn_done) begin
            res_valid <= 1'b1;
            res_logit <= logit;
            p_state   <= P_IDLE;
          end
        end
      endcase
    end
  end

  cnn_top #(.INIT_W(INIT_W), .INIT_B(INIT_B), .INIT_RQ(INIT_RQ)) u_cnn (
    .clk(clk), .rst_n(rst_n), .ld_valid(fb_valid), .ld_addr(fb_addr), .ld_q(fb_q), .start(cnn_start),
    .busy(cnn_busy), .done(cnn_done), .logit(logit));

  assign led_busy = cnn_busy;
  assign led_rec  = recording;
endmodule
