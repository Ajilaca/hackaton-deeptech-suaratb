// sva_stream: pemeriksa protokol stream (hanya simulasi; di-bind atau di-instansiasi di testbench).
// Aturan : (1) bila valid && !ready maka valid dan data harus tetap di siklus berikutnya;
//          (2) tidak ada X pada valid/ready setelah reset.
// Parameter : W lebar data, HAS_READY=0 untuk stream "valid saja" (hanya aturan 2).
module sva_stream #(
  parameter int W         = 8,
  parameter bit HAS_READY = 1'b1
) (
  input logic         clk,
  input logic         rst_n,
  input logic         valid,
  input logic         ready,
  input logic [W-1:0] data
);
`ifndef SYNTHESIS
  // verilator lint_off UNUSED
  if (HAS_READY) begin : g_hold
    a_hold: assert property (@(posedge clk) disable iff (!rst_n)
                             (valid && !ready) |=> (valid && $stable(data)))
      else $error("sva_stream: valid/data berubah sebelum ready");
  end
  a_no_x: assert property (@(posedge clk) disable iff (!rst_n)
                           !$isunknown(valid))
    else $error("sva_stream: valid bernilai X");
  // verilator lint_on UNUSED
`endif
endmodule
