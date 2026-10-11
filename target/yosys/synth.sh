#!/usr/bin/env bash
# Sintesis lintas-vendor Yosys (synth_xilinx -family xc7) untuk satu nilai P. Jalankan dari ROOT repo:
#   target/yosys/synth.sh 16     (butuh target/artix7/prep_synth.py --p 16 --out $SUARATB_DATA/build/synth/P16 lebih dulu)
# sv2v mengubah SystemVerilog -> Verilog (Yosys 0.33 belum membaca import paket/return fungsi). RTL tidak diubah.
set -eu
P=${1:?P}
D=${SUARATB_DATA:-/data/suaratb}
S=${SV2V:-/data/tools/sv2v/sv2v-Linux/sv2v}
B=$D/build/synth/P$P
O=$D/build/yosys/P$P
mkdir -p "$O"
files=$(grep -v '^//' rtl/files.f | grep -v sva_stream | sed "s#rtl/cnn/cnn_cfg_pkg.sv#$B/cnn_cfg_pkg.sv#" | tr '\n' ' ')
$S -DSYNTHESIS $files | sed "s#tb/vectors/cnn/#$B/#g" > "$O/top.v"
yosys -l "$O/yosys.log" -p "
read_verilog -sv $O/top.v
hierarchy -top suaratb_top
synth_xilinx -family xc7 -top suaratb_top -flatten
tee -o $O/stat.txt stat
write_verilog -noattr $O/netlist.v
" > /dev/null
echo "P=$P: $(grep -E 'Number of cells|LUT|DSP48|RAMB|FDRE|FDCE|FDPE' $O/stat.txt | tr -s ' ' | head -20 | tr '\n' ';')"
