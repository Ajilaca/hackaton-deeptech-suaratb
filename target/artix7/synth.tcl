# Sintesis + implementasi Vivado, Artix-7 XC7A35T-1 (referensi, bukan papan demo). Jalankan dari ROOT repo:
#   P=16 PDIR=/data/suaratb/build/synth/P16 OUT=/data/suaratb/build/vivado/P16 CLK_NS=10 vivado -mode batch -source target/artix7/synth.tcl
# PDIR berisi cnn_cfg_pkg.sv dan cnn_{wrom,bias,rq}.mem hasil target/artix7/prep_synth.py. Mode OOC: tanpa pin/MMCM.
set P      $::env(P)
set PDIR   $::env(PDIR)
set OUT    $::env(OUT)
set CLK_NS [expr {[info exists ::env(CLK_NS)] ? $::env(CLK_NS) : 10.0}]
set PART   [expr {[info exists ::env(PART)] ? $::env(PART) : "xc7a35tcpg236-1"}]
set IMPL   [expr {[info exists ::env(IMPL)] ? $::env(IMPL) : 1}]
file mkdir $OUT
set root [pwd]
set files {}
set fh [open rtl/files.f r]
while {[gets $fh line] >= 0} {
  set line [string trim $line]
  if {$line eq "" || [string match "//*" $line]} continue
  if {[string match "*sva_stream.sv" $line]} continue
  if {[string match "*cnn_cfg_pkg.sv" $line]} { lappend files $PDIR/cnn_cfg_pkg.sv; continue }
  lappend files $root/$line
}
close $fh
foreach f $files { read_verilog -sv $f }
set_param general.maxThreads 4
synth_design -top suaratb_top -part $PART -mode out_of_context -flatten_hierarchy rebuilt \
  -generic "INIT_W=\"$PDIR/cnn_wrom.mem\"" -generic "INIT_B=\"$PDIR/cnn_bias.mem\"" -generic "INIT_RQ=\"$PDIR/cnn_rq.mem\"" \
  -generic "INIT_WIN=\"$root/rtl/mem/window_hamming.mem\"" -generic "INIT_COS=\"$root/rtl/mem/twiddle_cos.mem\"" \
  -generic "INIT_SIN=\"$root/rtl/mem/twiddle_sin.mem\"" -generic "INIT_MEL=\"$root/rtl/mem/mel_rom.mem\"" \
  -generic "INIT_LOG=\"$root/rtl/mem/log_lut.mem\""
create_clock -name clk -period $CLK_NS [get_ports clk]
report_utilization -file $OUT/util_synth.rpt
report_timing_summary -file $OUT/timing_synth.rpt
write_checkpoint -force $OUT/synth.dcp
if {$IMPL} {
  opt_design
  place_design
  route_design
  report_utilization -file $OUT/util_impl.rpt
  report_utilization -hierarchical -file $OUT/util_impl_hier.rpt
  report_timing_summary -file $OUT/timing_impl.rpt
  report_power -file $OUT/power_impl.rpt
  write_checkpoint -force $OUT/impl.dcp
}
puts "SELESAI P=$P"
