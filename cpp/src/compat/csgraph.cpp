#include "csgraph.hpp"

namespace faqem {
namespace csg {

static i64 find(VecI& p, i64 x) {
    while (p[x] != x) {
        p[x] = p[p[x]];
        x = p[x];
    }
    return x;
}

i64 connected_components(i64 n, const VecI& src, const VecI& dst, VecI& labels) {
    VecI parent(n);
    for (i64 i = 0; i < n; i++) parent[i] = i;
    for (size_t e = 0; e < src.size(); e++) {
        i64 a = find(parent, src[e]), b = find(parent, dst[e]);
        if (a != b) {
            if (a < b) parent[b] = a;
            else parent[a] = b;
        }
    }
    labels.assign(n, -1);
    VecI root_label(n, -1);
    i64 nl = 0;
    for (i64 v = 0; v < n; v++) {
        i64 r = find(parent, v);
        if (root_label[r] < 0) root_label[r] = nl++;
        labels[v] = root_label[r];
    }
    return nl;
}

CSR build_csr(i64 n, const VecI& src, const VecI& dst) {
    CSR g;
    g.indptr.assign(n + 1, 0);
    for (i64 s : src) g.indptr[s + 1]++;
    for (i64 i = 0; i < n; i++) g.indptr[i + 1] += g.indptr[i];
    VecI fill(g.indptr.begin(), g.indptr.end() - 1);
    VecI cols(src.size());
    for (size_t e = 0; e < src.size(); e++) cols[fill[src[e]]++] = dst[e];
    // sort + dedupe each row
    CSR out;
    out.indptr.assign(n + 1, 0);
    for (i64 r = 0; r < n; r++) {
        auto b = cols.begin() + g.indptr[r], e = cols.begin() + g.indptr[r + 1];
        std::sort(b, e);
        auto last = std::unique(b, e);
        out.indices.insert(out.indices.end(), b, last);
        out.indptr[r + 1] = (i64)out.indices.size();
    }
    return out;
}

CSR transpose(i64 n, const CSR& g) {
    VecI src, dst;
    for (i64 r = 0; r < n; r++)
        for (i64 k = g.indptr[r]; k < g.indptr[r + 1]; k++) {
            src.push_back(g.indices[k]);
            dst.push_back(r);
        }
    return build_csr(n, src, dst);
}

void breadth_first_order(i64 n, const CSR& g, const CSR& gT, i64 head, VecI& nodes, VecI& pred) {
    pred.assign(n, -9999);
    VecI list(n);
    list[0] = head;
    i64 i_nl = 0, i_end = 1;
    while (i_nl < i_end) {
        i64 p = list[i_nl];
        for (const CSR* G : {&g, &gT}) {
            for (i64 k = G->indptr[p]; k < G->indptr[p + 1]; k++) {
                i64 c = G->indices[k];
                if (c == head) continue;
                if (pred[c] == -9999) {
                    list[i_end++] = c;
                    pred[c] = p;
                }
            }
        }
        i_nl++;
    }
    nodes.assign(list.begin(), list.begin() + i_end);
}

}  // namespace csg
}  // namespace faqem
