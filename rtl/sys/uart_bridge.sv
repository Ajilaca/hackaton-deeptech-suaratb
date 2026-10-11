// uart_bridge: protokol byte host <-> perangkat (lihat docs/vault/protokol-uart? ringkasan di bawah).
// Host -> perangkat:
//   'W' addr data      : tulis CSR[addr] = data
//   'R' addr           : balas 1 byte CSR[addr]
//   'D' + 8240 x int16 (LE) : klip dataset; ds_start lalu 8240 sampel (ds_valid/ds_sample, 1 per 2 byte)
// Perangkat -> host:
//   balasan 'R' (1 byte); hasil inferensi: 'L' lo hi seq (4 byte) setiap res_valid
// Parameter : CLK_PER_BIT, N_SAMPLES
// Catatan   : tanpa FIFO; jarak sampel dataset = 2 byte UART (>= 404 siklus butuh baud <= ~460800 pada 98,304 MHz).
module uart_bridge #(
  parameter int CLK_PER_BIT = 853,
  parameter int N_SAMPLES   = suaratb_pkg::CLIP_SAMPLES_PADDED
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        rx,
  output logic        tx,
  // bus CSR
  output logic        csr_we,
  output logic [7:0]  csr_waddr,
  output logic [7:0]  csr_wdata,
  output logic [7:0]  csr_raddr,
  input  logic [7:0]  csr_rdata,
  output logic        csr_rd,
  // aliran dataset
  output logic        ds_start,
  output logic        ds_valid,
  output logic signed [15:0] ds_sample,
  // hasil
  input  logic        res_valid,
  input  logic signed [15:0] res_logit,
  input  logic [7:0]  res_seq
);
  logic       rx_v;
  logic [7:0] rx_d;
  uart_rx #(.CLK_PER_BIT(CLK_PER_BIT)) u_rx (.clk(clk), .rst_n(rst_n), .rx(rx), .valid(rx_v), .data(rx_d));

  logic       tx_valid, tx_ready;
  logic [7:0] tx_data;
  uart_tx #(.CLK_PER_BIT(CLK_PER_BIT)) u_tx (.clk(clk), .rst_n(rst_n), .valid(tx_valid), .data(tx_data), .ready(tx_ready), .tx(tx));

  typedef enum logic [2:0] {S_CMD, S_W_ADDR, S_W_DATA, S_R_ADDR, S_D_LO, S_D_HI} st_t;
  st_t state;
  logic [7:0]  addr_q;
  logic [7:0]  lo_q;
  logic [15:0] n_left;

  // ---- penerima perintah + pengirim (satu blok) -----------------------------------------------------------
  logic        reply_pending;
  logic        reply_go;
  logic [7:0]  reply_byte;
  logic [7:0]  res_b [0:3];
  logic        res_busy;
  logic [1:0]  res_i;

  always_ff @(posedge clk) begin
    csr_we   <= 1'b0;
    csr_rd   <= 1'b0;
    ds_start <= 1'b0;
    ds_valid <= 1'b0;
    tx_valid <= 1'b0;
    if (!rst_n) begin
      state         <= S_CMD;
      reply_pending <= 1'b0;
      reply_go      <= 1'b0;
      res_busy      <= 1'b0;
      res_i         <= '0;
    end else begin
      // ---- perintah masuk
      if (rx_v) begin
        case (state)
          S_CMD: begin
            case (rx_d)
              8'h57: state <= S_W_ADDR;            // 'W'
              8'h52: state <= S_R_ADDR;            // 'R'
              8'h44: begin                         // 'D'
                state    <= S_D_LO;
                ds_start <= 1'b1;
                n_left   <= 16'(N_SAMPLES);
              end
              default: state <= S_CMD;
            endcase
          end
          S_W_ADDR: begin addr_q <= rx_d; state <= S_W_DATA; end
          S_W_DATA: begin
            csr_we    <= 1'b1;
            csr_waddr <= addr_q;
            csr_wdata <= rx_d;
            state     <= S_CMD;
          end
          S_R_ADDR: begin
            csr_raddr     <= rx_d;
            csr_rd        <= 1'b1;
            reply_pending <= 1'b1;                 // csr_rdata valid mulai siklus berikutnya
            state         <= S_CMD;
          end
          S_D_LO: begin lo_q <= rx_d; state <= S_D_HI; end
          S_D_HI: begin
            ds_valid  <= 1'b1;
            ds_sample <= {rx_d, lo_q};
            n_left    <= n_left - 16'd1;
            state     <= (n_left == 16'd1) ? S_CMD : S_D_LO;
          end
          default: state <= S_CMD;
        endcase
      end
      // ---- ambil nilai balasan 'R' (csr_raddr sudah stabil)
      if (reply_pending && !csr_rd) begin
        reply_byte    <= csr_rdata;
        reply_pending <= 1'b0;
        reply_go      <= 1'b1;
      end
      // ---- hasil inferensi
      if (res_valid) begin
        res_b[0] <= 8'h4C;                          // 'L'
        res_b[1] <= res_logit[7:0];
        res_b[2] <= res_logit[15:8];
        res_b[3] <= res_seq;
        res_busy <= 1'b1;
        res_i    <= '0;
      end
      // ---- pengirim: balasan R didahulukan, lalu bingkai hasil
      if (tx_ready && !tx_valid) begin
        if (reply_go) begin
          tx_valid <= 1'b1;
          tx_data  <= reply_byte;
          reply_go <= 1'b0;
        end else if (res_busy && !res_valid) begin
          tx_valid <= 1'b1;
          tx_data  <= res_b[res_i];
          res_i    <= res_i + 2'd1;
          if (res_i == 2'd3) res_busy <= 1'b0;
        end
      end
    end
  end
endmodule
