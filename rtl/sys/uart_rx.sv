// uart_rx: penerima UART 8N1 (LSB dulu), pencuplik di tengah bit.
// Parameter : CLK_PER_BIT (clk / baud, mis. 853 untuk 98,304 MHz / 115200)
// Antarmuka : rx (asinkron) -> valid (1 siklus), data
// Latensi   : 1,5 bit setelah tepi turun start bit + 1 siklus
module uart_rx #(
  parameter int CLK_PER_BIT = 853
) (
  input  logic       clk,
  input  logic       rst_n,
  input  logic       rx,
  output logic       valid,
  output logic [7:0] data
);
  localparam int CW = $clog2(CLK_PER_BIT + 1);
  typedef enum logic [1:0] {S_IDLE, S_START, S_DATA, S_STOP} st_t;
  st_t state;
  logic rx_m, rx_s;
  logic [CW-1:0] cnt;
  logic [2:0]    bit_i;
  logic [7:0]    sh;

  always_ff @(posedge clk) begin
    rx_m <= rx;
    rx_s <= rx_m;
    valid <= 1'b0;
    if (!rst_n) begin
      state <= S_IDLE;
      cnt   <= '0;
      bit_i <= '0;
    end else begin
      case (state)
        S_IDLE: begin
          if (!rx_s) begin
            state <= S_START;
            cnt   <= '0;
          end
        end
        S_START: begin                                  // tunggu tengah start bit
          if (cnt == CW'(CLK_PER_BIT / 2 - 1)) begin
            cnt <= '0;
            if (!rx_s) begin
              state <= S_DATA;
              bit_i <= '0;
            end else begin
              state <= S_IDLE;                          // glitch
            end
          end else begin
            cnt <= cnt + 1'b1;
          end
        end
        S_DATA: begin
          if (cnt == CW'(CLK_PER_BIT - 1)) begin
            cnt <= '0;
            sh  <= {rx_s, sh[7:1]};
            if (bit_i == 3'd7) state <= S_STOP;
            bit_i <= bit_i + 3'd1;
          end else begin
            cnt <= cnt + 1'b1;
          end
        end
        S_STOP: begin
          if (cnt == CW'(CLK_PER_BIT - 1)) begin
            cnt   <= '0;
            state <= S_IDLE;
            valid <= 1'b1;
            data  <= sh;
          end else begin
            cnt <= cnt + 1'b1;
          end
        end
      endcase
    end
  end
endmodule
