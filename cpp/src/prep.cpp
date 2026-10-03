#include "prep.hpp"

#include "compat/csgraph.hpp"

namespace faqem {

VecD PreparedMesh::to_model(const VecD& p) const {
    VecD out(p.size());
    for (size_t i = 0; i < p.size(); i += 3)
        for (int k = 0; k < 3; k++) out[i + k] = p[i + k] * diag + center[k];
    return out;
}

// manifold edge pairs: corner edges (f*3+k owns edge F[f,k] -> F[f,k+1]) sharing a 2-face edge
static void manifold_pairs(i64 nV, const VecI& F, VecI& e1, VecI& e2) {
    i64 m = (i64)F.size() / 3;
    VecI key(3 * m);
    for (i64 f = 0; f < m; f++)
        for (int k = 0; k < 3; k++) {
            i64 a = F[f * 3 + k], b = F[f * 3 + (k + 1) % 3];
            key[f * 3 + k] = std::min(a, b) * nV + std::max(a, b);
        }
    VecI order = np::argsort_stable(key);
    e1.clear();
    e2.clear();
    i64 n = (i64)order.size();
    i64 s = 0;
    while (s < n) {
        i64 e = s + 1;
        while (e < n && key[order[e]] == key[order[s]]) e++;
        if (e - s == 2) {
            e1.push_back(order[s]);
            e2.push_back(order[s + 1]);
        }
        s = e;
    }
}

static void split_non_manifold(const VecD& P, const VecI& F, VecD& P2, VecI& F2, VecI& grp_vertex) {
    i64 m = (i64)F.size() / 3;
    i64 nV = (i64)P.size() / 3;
    VecI e1, e2;
    manifold_pairs(nV, F, e1, e2);
    auto corner_of = [&](i64 f, i64 v) -> i64 {
        i64 c = F[f * 3] == v ? 0 : (F[f * 3 + 1] == v ? 1 : 2);
        return f * 3 + c;
    };
    size_t np_ = e1.size();
    VecI src(2 * np_), dst(2 * np_);
    for (size_t i = 0; i < np_; i++) {
        i64 c1 = e1[i], c2 = e2[i];
        i64 f1 = c1 / 3, f2 = c2 / 3;
        i64 u = F[c1], v = F[(c1 / 3) * 3 + (c1 % 3 + 1) % 3];
        src[i] = corner_of(f1, u);
        src[np_ + i] = corner_of(f1, v);
        dst[i] = corner_of(f2, u);
        dst[np_ + i] = corner_of(f2, v);
    }
    VecI label;
    i64 n_groups = csg::connected_components(3 * m, src, dst, label);
    grp_vertex.assign(n_groups, 0);
    for (i64 c = 0; c < 3 * m; c++) grp_vertex[label[c]] = F[c];
    F2 = label;
    P2.resize(n_groups * 3);
    for (i64 g = 0; g < n_groups; g++)
        for (int k = 0; k < 3; k++) P2[g * 3 + k] = P[grp_vertex[g] * 3 + k];
}

static i64 orient_consistently(const VecD& P, VecI& F) {
    i64 m = (i64)F.size() / 3;
    if (m == 0) return 0;
    i64 nV = (i64)P.size() / 3;
    VecI e1, e2;
    manifold_pairs(nV, F, e1, e2);
    size_t npair = e1.size();
    VecI f1(npair), f2(npair);
    std::vector<int8_t> parity(npair);
    bool any = false;
    for (size_t i = 0; i < npair; i++) {
        f1[i] = e1[i] / 3;
        f2[i] = e2[i] / 3;
        i64 a1 = F[e1[i]], b1 = F[(e1[i] / 3) * 3 + (e1[i] % 3 + 1) % 3];
        i64 a2 = F[e2[i]], b2 = F[(e2[i] / 3) * 3 + (e2[i] % 3 + 1) % 3];
        parity[i] = (a1 == a2) && (b1 == b2);
        any = any || parity[i];
    }
    if (!any) return 0;
    csg::CSR g = csg::build_csr(m, f1, f2);
    csg::CSR gT = csg::transpose(m, g);
    VecI comp;
    csg::connected_components(m, f1, f2, comp);
    VecI pk(npair);
    for (size_t i = 0; i < npair; i++) pk[i] = std::min(f1[i], f2[i]) * m + std::max(f1[i], f2[i]);
    VecI o = np::argsort_quick(pk);
    VecI pair_key(npair);
    std::vector<int8_t> pair_par(npair);
    for (size_t i = 0; i < npair; i++) {
        pair_key[i] = pk[o[i]];
        pair_par[i] = parity[o[i]];
    }
    VecD area(m);
    for (i64 f = 0; f < m; f++) {
        const double* a = &P[F[f * 3] * 3];
        const double* b = &P[F[f * 3 + 1] * 3];
        const double* c = &P[F[f * 3 + 2] * 3];
        double cx, cy, cz;
        np::cross(b[0] - a[0], b[1] - a[1], b[2] - a[2], c[0] - a[0], c[1] - a[1], c[2] - a[2], cx, cy, cz);
        area[f] = np::norm3(cx, cy, cz);
    }
    std::vector<int8_t> flip(m, 0);
    VecI bad;
    for (size_t i = 0; i < npair; i++)
        if (parity[i] == 1) bad.push_back(comp[f1[i]]);
    std::sort(bad.begin(), bad.end());
    bad.erase(std::unique(bad.begin(), bad.end()), bad.end());
    // members of each component, in index order
    for (i64 c : bad) {
        i64 seed = -1;
        double best = 0;
        for (i64 f = 0; f < m; f++) {
            if (comp[f] != c) continue;
            if (seed < 0 || area[f] > best) {  // argmax: first maximum
                seed = f;
                best = area[f];
            }
        }
        VecI nodes, pred;
        csg::breadth_first_order(m, g, gT, seed, nodes, pred);
        // _propagate_flips
        std::vector<int8_t> fl(m, 0);
        for (size_t i = 1; i < nodes.size(); i++) {
            i64 f = nodes[i];
            i64 p = pred[f];
            i64 lo = std::min(f, p), hi = std::max(f, p);
            i64 k = lo * m + hi;
            i64 j = np::searchsorted_left(pair_key, k);
            int8_t par = (j < (i64)npair && pair_key[j] == k) ? pair_par[j] : 0;
            fl[f] = fl[p] ^ par;
        }
        VecD a1, a0;
        std::vector<int8_t> fln(nodes.size());
        for (size_t i = 0; i < nodes.size(); i++) {
            fln[i] = fl[nodes[i]];
            (fln[i] == 1 ? a1 : a0).push_back(area[nodes[i]]);
        }
        if (np::sum(a1) > np::sum(a0))
            for (auto& x : fln) x ^= 1;
        for (size_t i = 0; i < nodes.size(); i++) flip[nodes[i]] = fln[i];
    }
    i64 nflip = 0;
    for (i64 f = 0; f < m; f++)
        if (flip[f]) {
            std::swap(F[f * 3], F[f * 3 + 2]);
            nflip++;
        }
    return nflip;
}

PreparedMesh prepare_mesh(const VecD& positions, const VecI& faces, double weld_tolerance, bool make_manifold) {
    PreparedMesh pm;
    i64 n = (i64)positions.size() / 3;
    i64 m0 = (i64)faces.size() / 3;
    double lo[3], hi[3];
    for (int k = 0; k < 3; k++) {
        lo[k] = positions[k];
        hi[k] = positions[k];
        for (i64 i = 1; i < n; i++) {
            lo[k] = np::fmin0(lo[k], positions[i * 3 + k]);
            hi[k] = np::fmax0(hi[k], positions[i * 3 + k]);
        }
    }
    double center[3], d[3];
    for (int k = 0; k < 3; k++) {
        center[k] = (lo[k] + hi[k]) / 2;
        d[k] = hi[k] - lo[k];
    }
    double diag = np::norm3(d[0], d[1], d[2]);
    if (diag == 0) diag = 1.0;
    VecD P(n * 3);
    for (i64 i = 0; i < n; i++)
        for (int k = 0; k < 3; k++) P[i * 3 + k] = (positions[i * 3 + k] - center[k]) / diag;

    // weld: quantise onto a grid of size tol
    VecI q(n * 3);
    for (i64 i = 0; i < n * 3; i++) q[i] = (i64)np::rint(P[i] / weld_tolerance);
    np::UniqueRows u = np::unique_rows(q.data(), n, 3);
    i64 nw = (i64)u.index.size();
    VecD welded(nw * 3);
    for (i64 i = 0; i < nw; i++)
        for (int k = 0; k < 3; k++) welded[i * 3 + k] = P[u.index[i] * 3 + k];
    VecI F(m0 * 3);
    for (i64 i = 0; i < m0 * 3; i++) F[i] = u.inverse[faces[i]];

    // drop degenerate + duplicate faces
    VecI idx;
    i64 n_degenerate = 0;
    for (i64 f = 0; f < m0; f++) {
        i64 a = F[f * 3], b = F[f * 3 + 1], c = F[f * 3 + 2];
        if (a != b && b != c && a != c) idx.push_back(f);
        else n_degenerate++;
    }
    VecI Fs(idx.size() * 3);
    for (size_t i = 0; i < idx.size(); i++) {
        i64 t[3] = {F[idx[i] * 3], F[idx[i] * 3 + 1], F[idx[i] * 3 + 2]};
        std::sort(t, t + 3);
        for (int k = 0; k < 3; k++) Fs[i * 3 + k] = t[k];
    }
    np::UniqueRows uf = np::unique_rows(Fs.data(), (i64)idx.size(), 3);
    VecI keep = uf.index;
    std::sort(keep.begin(), keep.end());
    i64 n_duplicate = (i64)idx.size() - (i64)keep.size();
    VecI face_orig(keep.size());
    for (size_t i = 0; i < keep.size(); i++) face_orig[i] = idx[keep[i]];
    i64 m = (i64)face_orig.size();
    VecI F1(m * 3);
    for (i64 i = 0; i < m; i++)
        for (int k = 0; k < 3; k++) F1[i * 3 + k] = F[face_orig[i] * 3 + k];

    // drop unreferenced vertices
    std::vector<char> used(nw, 0);
    for (i64 v : F1) used[v] = 1;
    VecI remap(nw, -1);
    i64 nu = 0;
    for (i64 i = 0; i < nw; i++)
        if (used[i]) remap[i] = nu++;
    for (auto& v : F1) v = remap[v];
    VecD out_pos(nu * 3);
    for (i64 i = 0; i < nw; i++)
        if (used[i])
            for (int k = 0; k < 3; k++) out_pos[remap[i] * 3 + k] = welded[i * 3 + k];
    VecI input_to_vertex(n);
    for (i64 i = 0; i < n; i++) input_to_vertex[i] = remap[u.inverse[i]];

    i64 n_split = 0;
    if (make_manifold) {
        VecD P2;
        VecI F2, src_vertex;
        split_non_manifold(out_pos, F1, P2, F2, src_vertex);
        n_split = (i64)P2.size() / 3 - nu;
        if (n_split > 0) {
            VecI first_copy(nu, -1);
            for (i64 g = (i64)src_vertex.size() - 1; g >= 0; g--) first_copy[src_vertex[g]] = g;
            for (auto& v : input_to_vertex) v = v >= 0 ? first_copy[v] : -1;
            out_pos = P2;
            F1 = F2;
        }
    }
    i64 n_flipped = orient_consistently(out_pos, F1);

    pm.positions = out_pos;
    pm.faces = F1;
    pm.face_orig = face_orig;
    pm.input_to_vertex = input_to_vertex;
    for (int k = 0; k < 3; k++) pm.center[k] = center[k];
    pm.diag = diag;
    pm.removed = {{"welded", n - nw},
                  {"degenerate", n_degenerate},
                  {"duplicate", n_duplicate},
                  {"split_non_manifold", n_split},
                  {"reoriented", n_flipped}};
    return pm;
}

void unique_edges(const VecI& F, i64 nV, VecI& edges, VecI& counts) {
    i64 m = (i64)F.size() / 3;
    VecI key(3 * m);
    // order of np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    for (int k = 0; k < 3; k++)
        for (i64 f = 0; f < m; f++) {
            i64 a = F[f * 3 + k], b = F[f * 3 + (k + 1) % 3];
            key[k * m + f] = std::min(a, b) * nV + std::max(a, b);
        }
    std::sort(key.begin(), key.end());
    edges.clear();
    counts.clear();
    for (size_t i = 0; i < key.size();) {
        size_t j = i + 1;
        while (j < key.size() && key[j] == key[i]) j++;
        edges.push_back(key[i] / nV);
        edges.push_back(key[i] % nV);
        counts.push_back((i64)(j - i));
        i = j;
    }
}

MeshStats mesh_stats(i64 nV, const VecI& F) {
    VecI edges, counts;
    unique_edges(F, nV, edges, counts);
    i64 ne = (i64)counts.size();
    VecI src(ne), dst(ne);
    for (i64 i = 0; i < ne; i++) {
        src[i] = edges[i * 2];
        dst[i] = edges[i * 2 + 1];
    }
    VecI lab;
    i64 nc = csg::connected_components(nV, src, dst, lab);
    MeshStats s{nV, (i64)F.size() / 3, ne, nc, 0, 0};
    for (i64 c : counts) {
        if (c == 1) s.boundary_edges++;
        if (c > 2) s.non_manifold_edges++;
    }
    return s;
}

}  // namespace faqem
