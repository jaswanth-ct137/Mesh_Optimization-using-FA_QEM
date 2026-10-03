// Triangle BVH with closest-point queries (port of faqem/bvh.py).
#pragma once
#include "compat/npcompat.hpp"

namespace faqem {

// Ericson RTCD 5.1.5. out = [b0, b1, b2, x, y, z]; returns squared distance.
// (inline: it is the innermost kernel of the local guarantee and of every BVH query)
inline double closest_on_tri(const double* P, const i64* F, i64 f, double px, double py, double pz, double* out) {
    i64 a = F[f * 3], b = F[f * 3 + 1], c = F[f * 3 + 2];
    double ax = P[a * 3], ay = P[a * 3 + 1], az = P[a * 3 + 2];
    double abx = P[b * 3] - ax, aby = P[b * 3 + 1] - ay, abz = P[b * 3 + 2] - az;
    double acx = P[c * 3] - ax, acy = P[c * 3 + 1] - ay, acz = P[c * 3 + 2] - az;
    double apx = px - ax, apy = py - ay, apz = pz - az;
    double d1 = abx * apx + aby * apy + abz * apz;
    double d2 = acx * apx + acy * apy + acz * apz;
    double v = 0.0, w = 0.0;
    if (d1 <= 0 && d2 <= 0) {
        v = 0.0;
        w = 0.0;
    } else {
        double bpx = px - P[b * 3], bpy = py - P[b * 3 + 1], bpz = pz - P[b * 3 + 2];
        double d3 = abx * bpx + aby * bpy + abz * bpz;
        double d4 = acx * bpx + acy * bpy + acz * bpz;
        if (d3 >= 0 && d4 <= d3) {
            v = 1.0;
            w = 0.0;
        } else {
            double vc = d1 * d4 - d3 * d2;
            if (vc <= 0 && d1 >= 0 && d3 <= 0) {
                double den = d1 - d3;
                v = den != 0 ? d1 / den : 0.0;
                w = 0.0;
            } else {
                double cpx = px - P[c * 3], cpy = py - P[c * 3 + 1], cpz = pz - P[c * 3 + 2];
                double d5 = abx * cpx + aby * cpy + abz * cpz;
                double d6 = acx * cpx + acy * cpy + acz * cpz;
                if (d6 >= 0 && d5 <= d6) {
                    v = 0.0;
                    w = 1.0;
                } else {
                    double vb = d5 * d2 - d1 * d6;
                    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
                        double den = d2 - d6;
                        w = den != 0 ? d2 / den : 0.0;
                        v = 0.0;
                    } else {
                        double va = d3 * d6 - d5 * d4;
                        if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
                            double den = (d4 - d3) + (d5 - d6);
                            w = den != 0 ? (d4 - d3) / den : 0.0;
                            v = 1.0 - w;
                        } else {
                            double den = va + vb + vc;
                            if (std::fabs(den) > 1e-300) {
                                v = vb / den;
                                w = vc / den;
                            }
                        }
                    }
                }
            }
        }
    }
    double x = ax + abx * v + acx * w;
    double y = ay + aby * v + acy * w;
    double z = az + abz * v + acz * w;
    out[0] = 1.0 - v - w;
    out[1] = v;
    out[2] = w;
    out[3] = x;
    out[4] = y;
    out[5] = z;
    double dx = x - px, dy = y - py, dz = z - pz;
    return dx * dx + dy * dy + dz * dz;
}


struct BVH {
    VecD P;      // (n, 3)
    VecI F;      // (m, 3)
    VecI idx;    // (m,)
    VecD box;    // (nodes, 6)
    VecI left, start, count;
    BVH() = default;
    BVH(const VecD& P_, const VecI& F_);
};

// Closest admissible triangle (see bvh.py). mode 0: any; 1: lineage filter; 2: lineage + facing.
// stack: >= 256 ints, sd: >= 256 doubles. Returns face (-1 if none) and d2 (via d2_out).
struct BVHView {
    const double* P;
    const i64* F;
    const i64* idx;
    const double* box;
    const i64* left;
    const i64* start;
    const i64* count;
};
inline BVHView view(const BVH& b) {
    return {b.P.data(), b.F.data(), b.idx.data(), b.box.data(), b.left.data(), b.start.data(), b.count.data()};
}
i64 bvh_closest(const BVHView& T, double px, double py, double pz, i64 hint, int mode, const i64* reps,
                i64 A, i64 B, i64 Cc, const double* fnrm, double nx, double ny, double nz, i64* stack,
                double* sd, int stack_len, double* out, double* tmp, double max_d2, double& d2_out);

}  // namespace faqem
