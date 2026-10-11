// rom_sync: ROM sinkron (1 siklus), vendor-neutral. Diinferensi sebagai block/distributed ROM.
// Parameter : WIDTH, DEPTH, INIT_FILE ($readmemh; baris '//' diabaikan)
// Antarmuka : clk, en, addr -> dout (valid 1 siklus setelah en)
// Latensi   : 1 siklus
// Golden    : tabel di golden/tables/*.mem (gen_tables.py)
module rom_sync #(
  parameter int    WIDTH     = 8,
  parameter int    DEPTH     = 256,
  parameter INIT_FILE = ""
) (
  input  logic                     clk,
  input  logic                     en,
  input  logic [$clog2(DEPTH)-1:0] addr,
  output logic [WIDTH-1:0]         dout
);
  logic [WIDTH-1:0] mem [0:DEPTH-1];

  initial begin
    if (INIT_FILE != "") $readmemh(INIT_FILE, mem);
  end

  always_ff @(posedge clk) begin
    if (en) dout <= mem[addr];
  end
endmodule
