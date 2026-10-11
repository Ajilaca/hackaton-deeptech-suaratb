// uart_tx: pengirim UART 8N1.
// Parameter : CLK_PER_BIT
// Antarmuka : valid/data (diterima hanya saat ready=1) -> tx; ready = idle
module uart_tx #(
  parameter int CLK_PER_BIT = 853
) (
  input  logic       clk,
  input  logic       rst_n,
  input  logic       valid,
  input  logic [7:0] data,
  output logic       ready,
  output logic       tx
);
  localparam int CW = $clog2(CLK_PER_BIT + 1);
  logic [9:0]    sh;          // {stop, data[7:0], start}
  logic [3:0]    n;           // bit tersisa
  logic [CW-1:0] cnt;

  assign ready = (n == 4'd0);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      n   <= 4'd0;
      tx  <= 1'b1;
      cnt <= '0;
    end else if (n == 4'd0) begin
      tx <= 1'b1;
      if (valid) begin
        sh  <= {1'b1, data, 1'b0};
        n   <= 4'd10;
        cnt <= '0;
      end
    end else begin
      tx <= sh[0];
      if (cnt == CW'(CLK_PER_BIT - 1)) begin
        cnt <= '0;
        sh  <= {1'b1, sh[9:1]};
        n   <= n - 4'd1;
      end else begin
        cnt <= cnt + 1'b1;
      end
    end
  end
endmodule
