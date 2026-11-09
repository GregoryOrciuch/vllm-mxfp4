#!/usr/bin/env python3
"""Generate a Python binding for Radiance 1.17's MXFP4 decode kernel.

Input is the libr4d directory at Radiance commit 19d4e8fe0e18071b9b6300f15fc4ccf38dc4fa3d.
Compile the generated file with hipcc and the input directory on the include path.
"""
import sys
from pathlib import Path

source = Path(sys.argv[1]) / "r4d_gemm_mxfp4a8_nt_m64.hip"
text = source.read_text()
include = '#include "r4d_gemm_args.h"\n'
marker = '// ============================================================== the plugin ABI entry points'
if text.count(include) != 1 or text.count(marker) != 1:
    raise SystemExit("unexpected Radiance kernel source; refusing to generate binding")
text = text.split(marker)[0].replace(include, "", 1)
text += '''
#include <pybind11/pybind11.h>
PYBIND11_MODULE(radiance117_mxfp4, m) {
  m.def("launch", [](long a, long ascale, long wq, long ws, long wref, long c,
                     int M, int K, int N, int WV, int SK, int MB, int NPW, long stream) {
    r4d_gemm_mxfp4a8_nt_m64(a, ascale, wq, ws, wref, c, M, K, N,
                             WV, SK, MB, NPW, 0, 1, stream, 0, 0, 0);
  });
}
'''
Path(sys.argv[2]).write_text(text)
