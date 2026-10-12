PY      ?= .venv/bin/python
SIM     ?= icarus
RTL_F   := rtl/files.f
# Sintesis (Fase 6): P = lane CNN, CLK_NS = periode clock (10 = 100 MHz, 6.667 = 150 MHz), MODEL = bobot int8 untuk ROM
P       ?= 16
CLK_NS  ?= 10
DATA    ?= $(or $(SUARATB_DATA),/data/suaratb)
MODEL   ?= $(DATA)/runs/model_v2/qmodel.npz

.PHONY: help golden tables vectors budget calib lint test test-unit test-golden synth-prep synth-yosys vivado wave clean-build

help:
	@echo "golden       : tabel .mem + paket SV + vektor sintetis dari golden/ (Fase 0)"
	@echo "tables       : salin tabel/paket ke rtl/ (rtl/mem, rtl/common/suaratb_pkg.sv)"
	@echo "test-golden  : tes pytest golden (Python saja, boleh kapan saja)"
	@echo "lint         : verilator --lint-only (HANYA bila diminta pengguna)"
	@echo "test-unit    : tes unit satu modul: make test-unit M=<modul> (mis. M=fft512)"
	@echo "test         : regresi RTL cocotb. HANYA Fase T: make test FASE_T=1"
	@echo "synth-yosys  : sintesis lintas-vendor Yosys: make synth-yosys P=16 (butuh sv2v dan yosys)"
	@echo "vivado       : sintesis+implementasi Vivado XC7A35T-1 OOC: make vivado P=16 CLK_NS=6.667"
	@echo "wave         : buka dump gelombang terakhir"

golden: tables vectors budget

tables:
	$(PY) -m golden.gen.gen_tables
	mkdir -p rtl/mem rtl/cnn
	cp golden/tables/cnn_cfg_pkg.sv rtl/cnn/cnn_cfg_pkg.sv
	cp golden/tables/suaratb_pkg.sv rtl/common/suaratb_pkg.sv
	cp golden/tables/*.mem rtl/mem/

vectors:
	$(PY) -m golden.gen.gen_vectors

budget:
	$(PY) -m golden.analysis.budget

calib:
	$(PY) -m golden.analysis.calibrate

test-golden:
	$(PY) -m pytest golden -q

lint:
	verilator --lint-only -Wall -Wno-fatal -Irtl/common -f $(RTL_F) --top-module $(TOP)

test-unit:
	SIM=$(SIM) $(PY) -m pytest $$(ls tb/*/test_$(M).py) -q -x

test:
ifndef FASE_T
	@echo "Ditolak: simulasi RTL hanya dijalankan di Fase T. Jalankan 'make test FASE_T=1'." && exit 1
endif
	SIM=$(SIM) $(PY) -m pytest tb -q

synth-yosys: synth-prep
	@mkdir -p $(DATA)/build && touch $(DATA)/build/.stamp_yosys
	SUARATB_DATA=$(DATA) target/yosys/synth.sh $(P)
	@test $(DATA)/build/yosys/P$(P)/stat.txt -nt $(DATA)/build/.stamp_yosys || { echo "Yosys gagal: lihat $(DATA)/build/yosys/P$(P)/yosys.log"; exit 1; }

vivado:
	@mkdir -p $(DATA)/build && touch $(DATA)/build/.stamp_vivado
	SUARATB_DATA=$(DATA) CLK_NS=$(CLK_NS) target/artix7/run_sweep.sh $(P)
	@test $(DATA)/build/vivado/P$(P)/timing_impl.rpt -nt $(DATA)/build/.stamp_vivado || { echo "Vivado gagal: lihat $(DATA)/build/vivado/P$(P).log"; exit 1; }
	@grep -A4 "WNS(ns)" $(DATA)/build/vivado/P$(P)/timing_impl.rpt | head -6

synth-prep:
	@test -f $(DATA)/build/synth/P$(P)/cnn_cfg_pkg.sv || $(PY) target/artix7/prep_synth.py --p $(P) --out $(DATA)/build/synth/P$(P) --model $(MODEL)

wave:
	gtkwave $$(ls -t target/build/*/*.vcd target/build/*/*.fst 2>/dev/null | head -1)

clean-build:
	rm -rf target/build/*
