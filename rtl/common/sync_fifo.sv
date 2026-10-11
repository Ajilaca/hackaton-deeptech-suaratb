// sync_fifo: FIFO sinkron valid/ready, DEPTH harus pangkat dua.
// Parameter : WIDTH, DEPTH
// Antarmuka : wr_valid/wr_ready/wr_data, rd_valid/rd_ready/rd_data (rd_data first-word-fall-through)
// Latensi   : 0 siklus (FWFT); baca kombinasional dari memori terdistribusi
module sync_fifo #(
  parameter int WIDTH = 8,
  parameter int DEPTH = 16
) (
  input  logic             clk,
  input  logic             rst_n,
  input  logic             wr_valid,
  output logic             wr_ready,
  input  logic [WIDTH-1:0] wr_data,
  output logic             rd_valid,
  input  logic             rd_ready,
  output logic [WIDTH-1:0] rd_data
);
  localparam int AW = $clog2(DEPTH);

  logic [WIDTH-1:0] mem [0:DEPTH-1];
  logic [AW:0]      wptr, rptr;

  wire empty = (wptr == rptr);
  wire full  = (wptr[AW] != rptr[AW]) && (wptr[AW-1:0] == rptr[AW-1:0]);

  assign wr_ready = !full;
  assign rd_valid = !empty;
  assign rd_data  = mem[rptr[AW-1:0]];

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      wptr <= '0;
      rptr <= '0;
    end else begin
      if (wr_valid && wr_ready) begin
        mem[wptr[AW-1:0]] <= wr_data;
        wptr <= wptr + 1'b1;
      end
      if (rd_valid && rd_ready) begin
        rptr <= rptr + 1'b1;
      end
    end
  end

`ifndef SYNTHESIS
  initial begin
    if ((DEPTH & (DEPTH - 1)) != 0) $error("sync_fifo: DEPTH harus pangkat dua");
  end
`endif
endmodule
