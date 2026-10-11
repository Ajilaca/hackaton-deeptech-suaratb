#!/usr/bin/env bash
# Sapuan P untuk Vivado (sintesis + implementasi OOC). Dari ROOT repo: target/artix7/run_sweep.sh [P ...]
set -u
D=${SUARATB_DATA:-/data/suaratb}
V=${VIVADO:-/data/tools/Xilinx/2026.1/Vivado/bin/vivado}
mkdir -p "$D/build/vivado"
for P in ${@:-16 8 32}; do
  [ -f "$D/build/synth/P$P/cnn_cfg_pkg.sv" ] || .venv/bin/python target/artix7/prep_synth.py --p "$P" --out "$D/build/synth/P$P" \
      --model "$D/runs/model_v2/qmodel.npz"
  P=$P PDIR=$D/build/synth/P$P OUT=$D/build/vivado/P$P CLK_NS=${CLK_NS:-10} \
    $V -mode batch -source target/artix7/synth.tcl -log "$D/build/vivado/P$P.log" -journal "$D/build/vivado/P$P.jou" > /dev/null 2>&1
  echo "P=$P rc=$? $(date +%H:%M)"
done
echo SELESAI
