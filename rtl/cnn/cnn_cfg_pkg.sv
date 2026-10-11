// Dihasilkan oleh golden/gen_cnn_rom.py dari params.layer_descs() (P=16). JANGAN diedit manual.
package cnn_cfg_pkg;
  localparam int NL = 5;
  localparam int WROM_DEPTH = 921;
  localparam int NBG = 7;
  localparam int GAP_CH = 32;
  localparam int GAP_HW = 30;
  localparam int GAP_RECIP = 1092;
  localparam int A_DEPTH = 4096;
  localparam int B_DEPTH = 8192;
  localparam int P = 16;
  function automatic int cfg_cin(input int l);
    case (l)
      0: return 1;
      1: return 16;
      2: return 32;
      3: return 32;
      4: return 16;
      default: return 0;
    endcase
  endfunction
  function automatic int cfg_cout(input int l);
    case (l)
      0: return 16;
      1: return 32;
      2: return 32;
      3: return 16;
      4: return 1;
      default: return 0;
    endcase
  endfunction
  function automatic int cfg_h(input int l);
    case (l)
      0: return 40;
      1: return 20;
      2: return 10;
      3: return 1;
      4: return 1;
      default: return 0;
    endcase
  endfunction
  function automatic int cfg_w(input int l);
    case (l)
      0: return 50;
      1: return 25;
      2: return 12;
      3: return 1;
      4: return 1;
      default: return 0;
    endcase
  endfunction
  function automatic int cfg_k(input int l);
    case (l)
      0: return 3;
      1: return 3;
      2: return 3;
      3: return 1;
      4: return 1;
      default: return 0;
    endcase
  endfunction
  function automatic int cfg_T(input int l);
    case (l)
      0: return 9;
      1: return 9;
      2: return 9;
      3: return 1;
      4: return 1;
      default: return 0;
    endcase
  endfunction
  function automatic int cfg_ho(input int l);
    case (l)
      0: return 20;
      1: return 10;
      2: return 5;
      3: return 1;
      4: return 1;
      default: return 0;
    endcase
  endfunction
  function automatic int cfg_wo(input int l);
    case (l)
      0: return 25;
      1: return 12;
      2: return 6;
      3: return 1;
      4: return 1;
      default: return 0;
    endcase
  endfunction
  function automatic int cfg_G(input int l);
    case (l)
      0: return 1;
      1: return 2;
      2: return 2;
      3: return 1;
      4: return 1;
      default: return 0;
    endcase
  endfunction
  function automatic int cfg_wbase(input int l);
    case (l)
      0: return 0;
      1: return 9;
      2: return 297;
      3: return 873;
      4: return 905;
      default: return 0;
    endcase
  endfunction
  function automatic int cfg_bbase(input int l);
    case (l)
      0: return 0;
      1: return 1;
      2: return 3;
      3: return 5;
      4: return 6;
      default: return 0;
    endcase
  endfunction
  function automatic int cfg_pool(input int l);
    case (l)
      0: return 1;
      1: return 1;
      2: return 1;
      3: return 0;
      4: return 0;
      default: return 0;
    endcase
  endfunction
  function automatic int cfg_final(input int l);
    case (l)
      0: return 0;
      1: return 0;
      2: return 0;
      3: return 0;
      4: return 1;
      default: return 0;
    endcase
  endfunction
  function automatic int cfg_src_b(input int l);
    case (l)
      0: return 0;
      1: return 1;
      2: return 0;
      3: return 0;
      4: return 1;
      default: return 0;
    endcase
  endfunction
  function automatic int cfg_dst_b(input int l);
    case (l)
      0: return 1;
      1: return 0;
      2: return 1;
      3: return 1;
      4: return 0;
      default: return 0;
    endcase
  endfunction
endpackage
