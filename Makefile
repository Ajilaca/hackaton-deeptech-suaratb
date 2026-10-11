PY      ?= .venv/bin/python
SIM     ?= icarus
RTL_F   := rtl/files.f

.PHONY: help golden tables vectors budget calib lint test test-unit test-golden synth-yosys vivado wave clean-build

help:
	@echo "golden       : tabel .mem + paket SV + vektor sintetis dari golden/ (Fase 0)"
	@echo "tables       : salin tabel/paket ke rtl/ (rtl/mem, rtl/common/suaratb_pkg.sv)"
	@echo "test-golden  : tes pytest golden (Python saja, boleh kapan saja)"
	@echo "lint         : verilator --lint-only (HANYA bila diminta pengguna)"
	@echo "test-unit    : tes unit satu modul: make test-unit M=<modul> (mis. M=fft512)"
	@echo "test         : regresi RTL cocotb. HANYA Fase T: make test FASE_T=1"
	@echo "synth-yosys  : sintesis lintas-vendor (Fase 6)"
	@echo "vivado       : sintesis Vivado XC7A35T (Fase 6)"
	@echo "wave         : buka dump gelombang terakhir"

golden: tables vectors budget

tables:
	$(PY) -m golden.gen_tables
	mkdir -p rtl/mem
	cp golden/tables/suaratb_pkg.sv rtl/common/suaratb_pkg.sv
	cp golden/tables/*.mem rtl/mem/

vectors:
	$(PY) -m golden.gen_vectors

budget:
	$(PY) -m golden.budget

calib:
	$(PY) -m golden.calibrate

test-golden:
	$(PY) -m pytest golden -q

lint:
	verilator --lint-only -Wall -Wno-fatal -Irtl/common -f $(RTL_F) --top-module $(TOP)

test-unit:
	SIM=$(SIM) $(PY) -m pytest tb/test_$(M).py -q -x

test:
ifndef FASE_T
	@echo "Ditolak: simulasi RTL hanya dijalankan di Fase T. Jalankan 'make test FASE_T=1'." && exit 1
endif
	SIM=$(SIM) $(PY) -m pytest tb -q

synth-yosys:
	@echo "Fase 6: belum ada skrip (target/yosys)."; exit 1

vivado:
	@echo "Fase 6: belum ada skrip (target/artix7)."; exit 1

wave:
	gtkwave $$(ls -t target/build/*/*.vcd target/build/*/*.fst 2>/dev/null | head -1)

clean-build:
	rm -rf target/build/*
