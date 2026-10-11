// ram_sdp: RAM dua-port sederhana (1 tulis, 1 baca sinkron), vendor-neutral.
// Parameter : WIDTH, DEPTH
// Antarmuka : tulis (we, waddr, wdata) dan baca (re, raddr -> rdata)
// Latensi   : baca 1 siklus; baca-saat-tulis alamat sama mengembalikan data LAMA (read-first)
module ram_sdp #(
  parameter int WIDTH = 8,
  parameter int DEPTH = 256
) (
  input  logic                     clk,
  input  logic                     we,
  input  logic [$clog2(DEPTH)-1:0] waddr,
  input  logic [WIDTH-1:0]         wdata,
  input  logic                     re,
  input  logic [$clog2(DEPTH)-1:0] raddr,
  output logic [WIDTH-1:0]         rdata
);
  logic [WIDTH-1:0] mem [0:DEPTH-1];

  always_ff @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
    if (re) rdata <= mem[raddr];
  end
endmodule
