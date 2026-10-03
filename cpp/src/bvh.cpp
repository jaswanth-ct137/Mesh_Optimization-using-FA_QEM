#include "bvh.hpp"

#include <limits>

namespace faqem {

static const double INF = std::numeric_limits<double>::infinity();

static void nth(i64* idx, const double* cen, int axis, i64 lo, i64 hi, i64 k) {
    while (hi > lo) {
        double pivot = cen[idx[(lo + hi) >> 1] * 3 + axis];
        i64 i = lo, j = hi;
        while (i <= j) {
            while (cen[idx[i] * 3 + axis] < pivot) i += 1;
            while (cen[idx[j] * 3 + axis] > pivot) j -= 1;
            if (i <= j) {
                std::swap(idx[i], idx[j]);
                i += 1;
                j -= 1;
            }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else return;
    }
}

BVH::BVH(const VecD& P_, const VecI& F_) : P(P_), F(F_) {
    const i64 leaf_size = 6;
    i64 nF = (i64)F.size() / 3;
    idx.resize(nF);
    for (i64 i = 0; i < nF; i++) idx[i] = i;
    VecD cen(nF * 3);
    for (i64 f = 0; f < nF; f++)
        for (int k = 0; k < 3; k++)
            cen[f * 3 + k] = (P[F[f * 3 + 0] * 3 + k] + P[F[f * 3 + 1] * 3 + k] + P[F[f * 3 + 2] * 3 + k]) / 3.0;
    i64 max_nodes = 2 * nF + 8;
    box.assign(max_nodes * 6, 0.0);
    left.assign(max_nodes, -1);
    start.assign(max_nodes, 0);
    count.assign(max_nodes, 0);
    std::vector<i64> stack(128 * 3);
    i64 sp = 0;
    stack[0] = 0;
    stack[1] = 0;
    stack[2] = nF;
    sp = 1;
    i64 n_nodes = 1;
    while (sp > 0) {
        sp -= 1;
        i64 node = stack[sp * 3], s = stack[sp * 3 + 1], e = stack[sp * 3 + 2];
        double lo0 = INF, lo1 = INF, lo2 = INF, hi0 = -INF, hi1 = -INF, hi2 = -INF;
        double c0l = INF, c1l = INF, c2l = INF, c0h = -INF, c1h = -INF, c2h = -INF;
        for (i64 i = s; i < e; i++) {
            i64 f = idx[i];
            for (int k = 0; k < 3; k++) {
                i64 v = F[f * 3 + k];
                double x = P[v * 3], y = P[v * 3 + 1], z = P[v * 3 + 2];
                lo0 = std::min(lo0, x);
                hi0 = std::max(hi0, x);
                lo1 = std::min(lo1, y);
                hi1 = std::max(hi1, y);
                lo2 = std::min(lo2, z);
                hi2 = std::max(hi2, z);
            }
            c0l = std::min(c0l, cen[f * 3]);
            c0h = std::max(c0h, cen[f * 3]);
            c1l = std::min(c1l, cen[f * 3 + 1]);
            c1h = std::max(c1h, cen[f * 3 + 1]);
            c2l = std::min(c2l, cen[f * 3 + 2]);
            c2h = std::max(c2h, cen[f * 3 + 2]);
        }
        double* bx = &box[node * 6];
        bx[0] = lo0;
        bx[1] = lo1;
        bx[2] = lo2;
        bx[3] = hi0;
        bx[4] = hi1;
        bx[5] = hi2;
        double ex = c0h - c0l, ey = c1h - c1l, ez = c2h - c2l;
        if (e - s <= leaf_size || std::max(std::max(ex, ey), ez) <= 0) {
            left[node] = -1;
            start[node] = s;
            count[node] = e - s;
            continue;
        }
        int axis = (ex >= ey && ex >= ez) ? 0 : (ey >= ez ? 1 : 2);
        i64 mid = (s + e) >> 1;
        nth(idx.data(), cen.data(), axis, s, e - 1, mid);
        i64 l = n_nodes;
        n_nodes += 2;
        left[node] = l;
        if (sp + 2 >= (i64)(stack.size() / 3)) stack.resize(stack.size() * 2);
        stack[sp * 3] = l;
        stack[sp * 3 + 1] = s;
        stack[sp * 3 + 2] = mid;
        sp++;
        stack[sp * 3] = l + 1;
        stack[sp * 3 + 1] = mid;
        stack[sp * 3 + 2] = e;
        sp++;
    }
    box.resize(n_nodes * 6);
    left.resize(n_nodes);
    start.resize(n_nodes);
    count.resize(n_nodes);
}

static inline double box_d2(const double* box, i64 n, double x, double y, double z) {
    const double* b = box + n * 6;
    double d = 0.0;
    double v = b[0] - x;
    if (v > 0) d += v * v;
    else {
        v = x - b[3];
        if (v > 0) d += v * v;
    }
    v = b[1] - y;
    if (v > 0) d += v * v;
    else {
        v = y - b[4];
        if (v > 0) d += v * v;
    }
    v = b[2] - z;
    if (v > 0) d += v * v;
    else {
        v = z - b[5];
        if (v > 0) d += v * v;
    }
    return d;
}

static inline bool admissible(i64 f, int mode, const i64* reps, i64 A, i64 B, i64 Cc, const double* fnrm,
                              double nx, double ny, double nz) {
    if (mode == 0) return true;
    bool ok = false;
    for (int k = 0; k < 3; k++) {
        i64 r = reps[f * 3 + k];
        if (r == A || r == B || r == Cc) {
            ok = true;
            break;
        }
    }
    if (!ok) return false;
    if (mode == 2) return fnrm[f * 3] * nx + fnrm[f * 3 + 1] * ny + fnrm[f * 3 + 2] * nz > 0.0;
    return true;
}

i64 bvh_closest(const BVHView& T, double px, double py, double pz, i64 hint, int mode, const i64* reps,
                i64 A, i64 B, i64 Cc, const double* fnrm, double nx, double ny, double nz, i64* stack,
                double* sd, int stack_len, double* out, double* tmp, double max_d2, double& d2_out) {
    double best = max_d2;
    i64 bf = -1;
    if (hint >= 0 && admissible(hint, mode, reps, A, B, Cc, fnrm, nx, ny, nz)) {
        double d2 = closest_on_tri(T.P, T.F, hint, px, py, pz, tmp);
        if (d2 < best) {
            best = d2;
            bf = hint;
            for (int k = 0; k < 6; k++) out[k] = tmp[k];
        }
    }
    i64 sp = 0;
    stack[0] = 0;
    sd[0] = box_d2(T.box, 0, px, py, pz);
    sp = 1;
    while (sp > 0) {
        sp -= 1;
        i64 node = stack[sp];
        if (sd[sp] >= best) continue;
        i64 l = T.left[node];
        if (l < 0) {
            i64 s = T.start[node];
            for (i64 i = s; i < s + T.count[node]; i++) {
                i64 f = T.idx[i];
                if (!admissible(f, mode, reps, A, B, Cc, fnrm, nx, ny, nz)) continue;
                double d2 = closest_on_tri(T.P, T.F, f, px, py, pz, tmp);
                if (d2 < best) {
                    best = d2;
                    bf = f;
                    for (int k = 0; k < 6; k++) out[k] = tmp[k];
                }
            }
            continue;
        }
        double dl = box_d2(T.box, l, px, py, pz);
        double dr = box_d2(T.box, l + 1, px, py, pz);
        if (sp + 2 >= stack_len) {
            d2_out = best;
            return bf;
        }
        if (dl < dr) {
            if (dr < best) {
                stack[sp] = l + 1;
                sd[sp] = dr;
                sp++;
            }
            if (dl < best) {
                stack[sp] = l;
                sd[sp] = dl;
                sp++;
            }
        } else {
            if (dl < best) {
                stack[sp] = l;
                sd[sp] = dl;
                sp++;
            }
            if (dr < best) {
                stack[sp] = l + 1;
                sd[sp] = dr;
                sp++;
            }
        }
    }
    d2_out = best;
    return bf;
}

}  // namespace faqem
