#include "metrics.hpp"

#include <atomic>
#include <limits>
#include <thread>

#include "compat/nprandom.hpp"
#include "parallel.hpp"

namespace faqem {

VecD sample_surface(const VecD& P, const VecI& F, i64 n, uint64_t seed, bool include_vertices) {
    i64 m = (i64)F.size() / 3, nV = (i64)P.size() / 3;
    VecD area(m);
    for (i64 f = 0; f < m; f++) {
        const double* a = &P[F[f * 3] * 3];
        const double* b = &P[F[f * 3 + 1] * 3];
        const double* c = &P[F[f * 3 + 2] * 3];
        double cx, cy, cz;
        np::cross(b[0] - a[0], b[1] - a[1], b[2] - a[2], c[0] - a[0], c[1] - a[1], c[2] - a[2], cx, cy, cz);
        area[f] = np::norm3(cx, cy, cz) / 2;
    }
    np::Generator rng(seed);
    double total = np::sum(area);
    if (!(total > 0)) return P;
    VecD p(m);
    for (i64 f = 0; f < m; f++) p[f] = area[f] / total;
    VecI fi = rng.choice(m, n, p);
    VecD u = rng.random(n), v = rng.random(n);
    for (i64 i = 0; i < n; i++) {
        if (u[i] + v[i] > 1) {
            u[i] = 1 - u[i];
            v[i] = 1 - v[i];
        }
    }
    VecD pts;
    pts.reserve(n * 3 + nV * 3);
    for (i64 i = 0; i < n; i++) {
        i64 f = fi[i];
        const double* a = &P[F[f * 3] * 3];
        const double* b = &P[F[f * 3 + 1] * 3];
        const double* c = &P[F[f * 3 + 2] * 3];
        for (int k = 0; k < 3; k++) pts.push_back(a[k] + u[i] * (b[k] - a[k]) + v[i] * (c[k] - a[k]));
    }
    if (include_vertices) {
        i64 step = std::max<i64>(1, nV / n);
        for (i64 i = 0; i < nV; i += step)
            for (int k = 0; k < 3; k++) pts.push_back(P[i * 3 + k]);
    }
    return pts;
}

VecD point_to_mesh_d2(const VecD& pts, const BVH& bvh) {
    i64 n = (i64)pts.size() / 3;
    VecD out(n);
    BVHView T = view(bvh);
    parallel_for(n, [&](i64 s, i64 e) {
        i64 stack[256];
        double sd[256], o[6], t[6];
        i64 reps[3] = {0, 0, 0};
        double fn[3] = {0, 0, 0};
        for (i64 i = s; i < e; i++) {
            double d2;
            bvh_closest(T, pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2], -1, 0, reps, -1, -1, -1, fn, 0.0, 0.0, 0.0,
                        stack, sd, 256, o, t, std::numeric_limits<double>::infinity(), d2);
            out[i] = d2;
        }
    });
    return out;
}

Metrics compare_meshes(const VecD& PA, const VecI& FA, const VecD& PB, const VecI& FB, i64 samples,
                       const BVH* bvh_a) {
    BVH own_a;
    if (!bvh_a) {
        own_a = BVH(PA, FA);
        bvh_a = &own_a;
    }
    BVH B(PB, FB);
    VecD sa = sample_surface(PA, FA, samples, 7);
    VecD sb = sample_surface(PB, FB, samples, 11);
    VecD dab = point_to_mesh_d2(sa, B);
    VecD dba = point_to_mesh_d2(sb, *bvh_a);
    Metrics m;
    double ma = np::vmax(dab.data(), (i64)dab.size()), mb = np::vmax(dba.data(), (i64)dba.size());
    m.hausdorff = std::sqrt(mb > ma ? mb : ma);
    m.chamfer = np::mean(dab) + np::mean(dba);
    VecD ra(dab.size()), rb(dba.size());
    for (size_t i = 0; i < dab.size(); i++) ra[i] = std::sqrt(dab[i]);
    for (size_t i = 0; i < dba.size(); i++) rb[i] = std::sqrt(dba[i]);
    m.mean_distance = (np::mean(ra) + np::mean(rb)) / 2;
    return m;
}

}  // namespace faqem
