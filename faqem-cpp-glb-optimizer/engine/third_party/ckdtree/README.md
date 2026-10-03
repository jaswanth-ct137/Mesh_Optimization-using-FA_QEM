scipy 1.15.0 `scipy/spatial/ckdtree/src` (BSD-3-Clause, see LICENSE.txt), with one marked patch in `build.cxx` (FAQEM_PARTITION / FAQEM_NTH_ELEMENT:
stable variants in the portable build), fetched from
https://github.com/scipy/scipy/tree/v1.15.0/scipy/spatial/ckdtree/src. `numpy/npy_common.h` is a
three-macro stand-in for the numpy header. Used by `src/virtual_edges.cpp` so cKDTree ball queries
(and their tie order) match the Python reference exactly.
