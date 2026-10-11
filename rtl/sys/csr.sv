// csr: file register 8 bit (alamat 8 bit) untuk kontrol/status perangkat lewat UART.
// Peta     : 0x00 CTRL  (bit0 mode: 0=dataset 1=mikrofon; bit1 clear: tulis 1 -> denyut bersih-bersih)
//            0x01 STATUS (R): bit0 cnn_busy, bit1 recording, bit2 result_new (hapus saat baca LOGIT_H)
//            0x02 LOGIT_L, 0x03 LOGIT_H (R)  : logit terakhir (int16)
//            0x04 SEQ (R)                    : jumlah hasil (modulo 256)
//            0x05 NSAT_L, 0x06 NSAT_H (R)    : maks jumlah saturasi FFT per frame sejak clear
//            0x07 VERSION (R)                : 0x01
//            0x08 FRAMES_L, 0x09 FRAMES_H (R): jumlah frame fitur (feat_buf.frame_count[15:0])
// Antarmuka : bus tulis/baca satu siklus (rdata kombinasional)
module csr (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        we,
  input  logic [7:0]  waddr,
  input  logic [7:0]  wdata,
  input  logic [7:0]  raddr,
  output logic [7:0]  rdata,
  input  logic        rd_strobe,             // baca terjadi (untuk menghapus result_new)
  // masukan perangkat keras
  input  logic        cnn_busy,
  input  logic        recording,
  input  logic        res_valid,
  input  logic signed [15:0] res_logit,
  input  logic [15:0] nsat_frame,            // n_sat satu frame (denyut saat nsat_valid)
  input  logic        nsat_valid,
  input  logic [15:0] frame_count,
  // keluaran kontrol
  output logic        mode_mic,
  output logic        clear_pulse,
  output logic [7:0]  seq
);
  logic        mic_en_r, res_new;
  logic [15:0] logit_r, nsat_max;

  assign mode_mic = mic_en_r;

  always_ff @(posedge clk) begin
    clear_pulse <= 1'b0;
    if (!rst_n) begin
      mic_en_r <= 1'b0;
      res_new  <= 1'b0;
      logit_r  <= '0;
      nsat_max <= '0;
      seq      <= '0;
    end else begin
      if (we && waddr == 8'h00) begin
        mic_en_r <= wdata[0];
        if (wdata[1]) begin
          clear_pulse <= 1'b1;
          nsat_max    <= '0;
          res_new     <= 1'b0;
        end
      end
      if (res_valid) begin
        logit_r <= res_logit;
        seq     <= seq + 8'd1;
        res_new <= 1'b1;
      end
      if (nsat_valid && nsat_frame > nsat_max) nsat_max <= nsat_frame;
      if (rd_strobe && raddr == 8'h03) res_new <= 1'b0;
    end
  end

  always_comb begin
    case (raddr)
      8'h00: rdata = {6'd0, 1'b0, mic_en_r};
      8'h01: rdata = {5'd0, res_new, recording, cnn_busy};
      8'h02: rdata = logit_r[7:0];
      8'h03: rdata = logit_r[15:8];
      8'h04: rdata = seq;
      8'h05: rdata = nsat_max[7:0];
      8'h06: rdata = nsat_max[15:8];
      8'h07: rdata = 8'h01;
      8'h08: rdata = frame_count[7:0];
      8'h09: rdata = frame_count[15:8];
      default: rdata = 8'h00;
    endcase
  end
endmodule
