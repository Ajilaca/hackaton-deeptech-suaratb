// cough_det: detektor batuk berbasis energi (EMA noise floor + rasio ambang + refractory).
// Fungsi    : per masukan e (satu per frame, e_valid):
//               nq0 = first ? e<<S : nq ; N = max(nq0>>>S, N_FLOOR) ; rec = (f <= cap_end)
//               nq  <= rec ? nq0 : nq0 + (((e<<S) - nq0) >>> S)
//               above = e > K*N ; block = rec || (f <= dead_until)
//               run   = block ? 0 : (above ? run+1 : 0)
//               run >= MIN_HOPS -> trig, onset = f-MIN_HOPS+1, start = max(onset-PRE_ROLL,0),
//                                  cap_end = onset+POST_ONSET-1, dead_until = cap_end+HOLD_HOPS, run = 0
//             seg_done berdenyut pada frame f == cap_end (segmen 50 frame lengkap di feat_buf).
// Parameter : K, EMA_SHIFT, MIN_HOPS, HOLD_HOPS, N_FLOOR, PRE_ROLL, POST_ONSET (default dari suaratb_pkg), E_W
// Antarmuka : e_valid/e -> trig/onset_frame/start_frame/end_frame, seg_done, recording
// Latensi   : 6 siklus setelah e_valid (tangkap e; nq_cur; N dan EMA; K*N dan jendela refractory; banding; perbarui);
//             e_valid berikutnya harus >= 6 siklus kemudian (di sistem: 1 per frame, >= ribuan siklus)
// Golden    : golden/blocks/cough_det.py : cough_det(E) (urutan dan aritmetika identik; floor pada >>>)
module cough_det #(
  parameter int K          = suaratb_pkg::DET_K,
  parameter int EMA_SHIFT  = suaratb_pkg::DET_EMA_SHIFT,
  parameter int MIN_HOPS   = suaratb_pkg::DET_MIN_HOPS,
  parameter int HOLD_HOPS  = suaratb_pkg::DET_HOLD_HOPS,
  parameter int N_FLOOR    = suaratb_pkg::DET_N_FLOOR,
  parameter int PRE_ROLL   = suaratb_pkg::PRE_ROLL,
  parameter int POST_ONSET = suaratb_pkg::POST_ONSET,
  parameter int E_W        = 48
) (
  input  logic             clk,
  input  logic             rst_n,
  input  logic             e_valid,
  input  logic [E_W-1:0]   e,
  output logic             trig,
  output logic [31:0]      onset_frame,
  output logic [31:0]      start_frame,
  output logic [31:0]      end_frame,
  output logic             seg_done,
  output logic             recording
);
  localparam int NW = E_W + EMA_SHIFT + 2;                 // lebar aritmetik nq (e << EMA_SHIFT tidak melimpah)
  localparam int CW = 34;                                  // lebar penghitung frame bertanda (f 32 bit + HOLD)
  localparam logic signed [NW-1:0] K_S       = NW'(K);
  localparam logic signed [NW-1:0] N_FLOOR_S = NW'(N_FLOOR);

  logic signed [NW-1:0] nq;
  logic                 first;
  logic [31:0]          f;
  logic signed [CW-1:0] cap_end, dead_until;
  logic [15:0]          run;

  // v1..v5: tahap alur; e_q ditahan sampai v4
  logic                 v1, v2, v3, v4, v5;
  logic [E_W-1:0]       e_q;
  logic signed [NW-1:0] nq_cur_q, esh_q, nq_upd_q, n_q, kn_q;
  logic                 isrec_q, dead_q, above_q, block_q;

  wire signed [NW-1:0] e_s   = $signed({{(NW-E_W){1'b0}}, e_q});
  wire signed [NW-1:0] e_sh  = e_s <<< EMA_SHIFT;
  wire signed [CW-1:0] f_s   = $signed({{(CW-32){1'b0}}, f});

  wire signed [NW-1:0] n_raw = nq_cur_q >>> EMA_SHIFT;
  wire signed [NW-1:0] n_cur = (n_raw > N_FLOOR_S) ? n_raw : N_FLOOR_S;
  wire signed [NW-1:0] nq_upd = nq_cur_q + ((esh_q - nq_cur_q) >>> EMA_SHIFT);

  wire [15:0] run_n  = above_q ? (run + 16'd1) : 16'd0;
  wire        detect = !block_q && (run_n >= 16'(MIN_HOPS));

  wire [31:0] onset_n  = f - 32'(MIN_HOPS) + 32'd1;
  wire [31:0] start_n  = (onset_n >= 32'(PRE_ROLL)) ? (onset_n - 32'(PRE_ROLL)) : 32'd0;
  wire [31:0] end_n    = onset_n + 32'(POST_ONSET) - 32'd1;
  wire signed [CW-1:0] end_s = $signed({{(CW-32){1'b0}}, end_n});

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      nq          <= '0;
      first       <= 1'b1;
      f           <= '0;
      cap_end     <= -CW'(1);
      dead_until  <= -CW'(1);
      run         <= '0;
      v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0; v5 <= 1'b0;
      trig        <= 1'b0;
      seg_done    <= 1'b0;
      recording   <= 1'b0;
      onset_frame <= '0;
      start_frame <= '0;
      end_frame   <= '0;
    end else begin
      trig     <= 1'b0;
      seg_done <= 1'b0;
      v1 <= e_valid;
      v2 <= v1;
      v3 <= v2;
      v4 <= v3;
      v5 <= v4;
      if (e_valid) e_q <= e;
      // v1: nq_cur = first ? e<<S : nq
      if (v1) begin
        nq_cur_q <= first ? e_sh : nq;
        esh_q    <= e_sh;
      end
      // v2: N (floor) dan pembaruan EMA
      if (v2) begin
        n_q      <= n_cur;
        nq_upd_q <= nq_upd;
      end
      // v3: K*N dan penanda jendela (rekam / refractory)
      if (v3) begin
        kn_q    <= K_S * n_q;
        isrec_q <= (f_s <= cap_end);
        dead_q  <= (f_s <= dead_until);
      end
      // v4: banding energi dengan K*N
      if (v4) begin
        above_q <= (e_s > kn_q);
        block_q <= isrec_q || dead_q;
      end
      // v5: perbarui keadaan
      if (v5) begin
        first     <= 1'b0;
        f         <= f + 32'd1;
        nq        <= isrec_q ? nq_cur_q : nq_upd_q;
        recording <= isrec_q;
        seg_done  <= isrec_q && (f_s == cap_end);
        if (block_q || detect) run <= '0;
        else                   run <= run_n;
        if (detect) begin
          trig        <= 1'b1;
          onset_frame <= onset_n;
          start_frame <= start_n;
          end_frame   <= end_n;
          cap_end     <= end_s;
          dead_until  <= end_s + CW'(HOLD_HOPS);
        end
      end
    end
  end
endmodule
