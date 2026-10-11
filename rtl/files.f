// Daftar berkas RTL (urutan = urutan kompilasi). Tambahkan modul baru di bagian bawah per fase.
rtl/common/suaratb_pkg.sv
rtl/common/rom_sync.sv
rtl/common/ram_sdp.sv
rtl/common/sync_fifo.sv
rtl/common/sva_stream.sv
// Fase 2: front-end
rtl/audio/i2s_rx.sv
rtl/audio/preemph.sv
rtl/audio/frame_buf.sv
rtl/detect/cough_det.sv
// Fase 3: DSP
rtl/dsp/window.sv
rtl/dsp/fft512.sv
rtl/dsp/power.sv
rtl/dsp/mel.sv
rtl/dsp/log_q.sv
rtl/dsp/quant.sv
rtl/dsp/dsp_chain.sv
rtl/dsp/feat_buf.sv
