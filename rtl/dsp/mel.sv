// mel: filterbank mel 40 band (HTK, 50-8000 Hz), bobot Q0.12 sparse dalam ROM, 40 akumulator.
// Fungsi    : tiap bin k (0..256, berurutan) diberi daya P[k]. ROM sparse berisi kata {band[5:0], w[11:0], eob} berurutan
//             (maks. 2 kata per bin; bin tanpa band punya satu kata bobot 0). Untuk tiap kata: acc[band] += P[k] * w.
//             Setelah bin terakhir (in_last), 40 akumulator dikeluarkan berurutan (1/siklus), E_b = min(acc >> MEL_SHIFT,
//             2^OUT_BITS - 1), lalu dikosongkan.
// Parameter : P_W (2W), ACC_W (MEL_ACC_BITS), OUT_BITS (LOG_IN_BITS), MEL_SHIFT, N_MEL, DEPTH (kata ROM), INIT_FILE
// Antarmuka : in_valid/in_p/in_first/in_last (bin berurutan; jarak antar bin >= 3 siklus)
//             -> out_valid/out_e/out_idx/out_last (40 keluaran berurutan)
// Latensi   : 2-3 siklus per bin; keluaran dimulai ~4 siklus setelah bin terakhir
// Golden    : golden/blocks/mel.py : mel_energy(p, shift, in_bits)
module mel #(
  parameter int    P_W       = 2 * suaratb_pkg::FFT_W,
  parameter int    ACC_W     = suaratb_pkg::MEL_ACC_BITS,
  parameter int    OUT_BITS  = suaratb_pkg::LOG_IN_BITS,
  parameter int    MEL_SHIFT = suaratb_pkg::MEL_SHIFT,
  parameter int    N_MEL     = suaratb_pkg::N_MEL,
  parameter int    W_FRAC    = suaratb_pkg::MEL_W_FRAC,
  parameter int    DEPTH     = 493,
  parameter INIT_FILE = "rtl/mem/mel_rom.mem"
) (
  input  logic                clk,
  input  logic                rst_n,
  input  logic                in_valid,
  input  logic [P_W-1:0]      in_p,
  input  logic                in_first,
  input  logic                in_last,
  output logic                out_valid,
  output logic [OUT_BITS-1:0] out_e,
  output logic [5:0]          out_idx,
  output logic                out_last
);
  localparam int AW   = $clog2(DEPTH);
  localparam int MW   = P_W + W_FRAC;               // lebar hasil kali

  typedef enum logic [1:0] {S_IDLE, S_WORD, S_DRAIN} state_t;
  state_t state;

  logic [AW-1:0]       ptr;                          // alamat kata ROM berikutnya
  logic [P_W-1:0]      p_hold;
  logic                last_hold;
  logic                rom_en;
  logic [AW-1:0]       rom_addr;
  logic [5+W_FRAC+1:0] rom_word;                      // {band[5:0], w[W_FRAC-1:0], eob}

  rom_sync #(.WIDTH(6 + W_FRAC + 1), .DEPTH(DEPTH), .INIT_FILE(INIT_FILE)) u_rom (
    .clk(clk), .en(rom_en), .addr(rom_addr), .dout(rom_word)
  );

  wire [5:0]        w_band = rom_word[W_FRAC+6:W_FRAC+1];
  wire [W_FRAC-1:0] w_val  = rom_word[W_FRAC:1];
  wire              w_eob  = rom_word[0];

  // ---- akumulator ----------------------------------------------------------------------------------
  logic [ACC_W-1:0] acc [0:N_MEL-1];
  logic             m_valid;                          // produk valid (tahap kali)
  logic [MW-1:0]    m_prod;
  logic [5:0]       m_band;
  logic             fin1, fin2;                        // bin terakhir: tandai akhir frame
  logic             out_run;
  logic [5:0]       o_cnt;

  // kontrol bin
  always_comb begin
    rom_en   = 1'b0;
    rom_addr = ptr;
    if (state == S_IDLE && in_valid) begin
      rom_en   = 1'b1;
      rom_addr = in_first ? '0 : ptr;
    end else if (state == S_WORD && !w_eob) begin
      rom_en = 1'b1;                                  // kata berikutnya dari bin yang sama
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state     <= S_IDLE;
      ptr       <= '0;
      m_valid   <= 1'b0;
      fin1      <= 1'b0;
      fin2      <= 1'b0;
      out_run   <= 1'b0;
      o_cnt     <= '0;
      out_valid <= 1'b0;
      out_last  <= 1'b0;
      for (int b = 0; b < N_MEL; b++) acc[b] <= '0;
    end else begin
      m_valid   <= 1'b0;
      fin1      <= 1'b0;
      out_valid <= 1'b0;
      out_last  <= 1'b0;
      fin2      <= fin1;

      case (state)
        S_IDLE: begin
          if (in_valid) begin
            p_hold    <= in_p;
            last_hold <= in_last;
            ptr       <= (in_first ? '0 : ptr) + 1'b1;   // ROM dibaca di alamat ini; kata berikutnya = +1
            state     <= S_WORD;
          end
        end
        S_WORD: begin
          // rom_word valid: kata untuk bin ini
          m_valid <= 1'b1;
          m_prod  <= p_hold * w_val;
          m_band  <= w_band;
          if (w_eob) begin
            fin1  <= last_hold;
            state <= S_IDLE;
          end else begin
            ptr <= ptr + 1'b1;
          end
        end
        default: state <= S_IDLE;
      endcase

      // akumulasi (tahap setelah kali)
      if (m_valid) acc[m_band] <= acc[m_band] + ACC_W'(m_prod);

      // keluaran: mulai setelah akumulasi kata terakhir selesai
      if (fin2 && !out_run) begin
        out_run <= 1'b1;
        o_cnt   <= '0;
      end
      if (out_run) begin
        out_valid <= 1'b1;
        out_idx   <= o_cnt;
        out_last  <= (o_cnt == 6'(N_MEL - 1));
        out_e     <= sat_out(acc[o_cnt]);
        acc[o_cnt] <= '0;
        if (o_cnt == 6'(N_MEL - 1)) out_run <= 1'b0;
        else                        o_cnt <= o_cnt + 6'd1;
      end
    end
  end

  function automatic logic [OUT_BITS-1:0] sat_out(input logic [ACC_W-1:0] a);
    logic [ACC_W-1:0] s;
    s = a >> MEL_SHIFT;
    if (ACC_W - MEL_SHIFT > OUT_BITS && (s >> OUT_BITS) != '0) sat_out = {OUT_BITS{1'b1}};
    else sat_out = s[OUT_BITS-1:0];
  endfunction

`ifndef SYNTHESIS
  always_ff @(posedge clk) begin
    if (rst_n && in_valid && state != S_IDLE)
      $error("mel: bin baru datang saat bin sebelumnya belum selesai (jarak < 3 siklus)");
  end
`endif
endmodule
