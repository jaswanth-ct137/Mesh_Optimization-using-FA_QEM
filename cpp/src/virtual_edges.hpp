// Virtual edge insertion (port of faqem/virtual_edges.py).
#pragma once
#include "compat/npcompat.hpp"

namespace faqem {

// returns (k, 2) vertex pairs (row-major)
VecI find_virtual_edges(const VecD& P, const VecI& F, double tau = 0.01, i64 max_components = 4000);

}  // namespace faqem
