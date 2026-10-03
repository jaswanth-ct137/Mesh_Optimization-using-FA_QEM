// scipy.sparse.csgraph (1.15.0) replicas with identical output ordering.
#pragma once
#include "npcompat.hpp"

namespace faqem {
namespace csg {

// connected_components(coo_matrix((1, (src, dst)), (n, n)), directed=False):
// labels are numbered in order of each component's smallest node.
i64 connected_components(i64 n, const VecI& src, const VecI& dst, VecI& labels);

// CSR with sorted, de-duplicated column indices (coo -> csr -> sum_duplicates)
struct CSR {
    VecI indptr, indices;
};
CSR build_csr(i64 n, const VecI& src, const VecI& dst);
CSR transpose(i64 n, const CSR& g);

// breadth_first_order(g, seed, directed=False, return_predecessors=True)
// pred: -9999 for unreached nodes
void breadth_first_order(i64 n, const CSR& g, const CSR& gT, i64 seed, VecI& nodes, VecI& pred);

}  // namespace csg
}  // namespace faqem
