#include "search.hpp"

#include <chrono>
#include <cmath>

namespace faqem {

static inline double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// Python builtins min(a, b) / max(a, b) on floats
static inline double pmin(double a, double b) { return b < a ? b : a; }
static inline double pmax(double a, double b) { return b > a ? b : a; }

const std::map<std::string, double>& detail_levels() {
    static const std::map<std::string, double> levels = {
        {"Low", 0.005}, {"Medium", 0.002}, {"High", 0.001}, {"Ultra", 0.0005}};
    return levels;
}

SearchResult simplify_auto(const PreparedMesh& prep, const BVH& bvh, double tolerance, const Options& options,
                           int max_passes, const PassProgress& progress) {
    double t0 = now();
    Options o = options;
    o.auto_mode = true;
    SearchResult out;
    bool have_ok = false, have_any = false;
    SimplifyResult best_ok_res, best_any_res;
    Metrics best_ok_m, best_any_m;
    double f = 0.8, f_ok = 0, f_bad = 0;
    bool has_f_ok = false, has_f_bad = false;
    for (int k = 0; k < max_passes; k++) {
        o.local_tol = f * tolerance;
        o.max_error = f * tolerance;
        SimplifyResult res = simplify(prep.positions, prep.faces, 1, o, &bvh);
        Metrics m = compare_meshes(prep.positions, prep.faces, res.positions, res.faces, 50000, &bvh);
        i64 faces = res.output_faces();
        double H = m.hausdorff;
        bool ok = H <= tolerance;
        out.passes.push_back({f, faces, H, ok});
        if (progress) progress(k, faces);
        bool take_ok = ok && (!have_ok || faces < best_ok_res.output_faces());
        bool take_any = !have_any || H < best_any_m.hausdorff;
        if (take_ok && take_any) {
            best_ok_res = res;
            best_ok_m = m;
            best_any_res = std::move(res);
            best_any_m = m;
            have_ok = have_any = true;
        } else if (take_ok) {
            best_ok_res = std::move(res);
            best_ok_m = m;
            have_ok = true;
        } else if (take_any) {
            best_any_res = std::move(res);
            best_any_m = m;
            have_any = true;
        }
        if (ok) {
            f_ok = has_f_ok ? pmax(f_ok, f) : f;
            has_f_ok = true;
            if (H > 0.85 * tolerance || f >= 1.0) break;
        } else {
            f_bad = has_f_bad ? pmin(f_bad, f) : f;
            has_f_bad = true;
        }
        if (has_f_ok && has_f_bad) {
            if (f_bad / f_ok < 1.12) break;
            f = std::sqrt(f_ok * f_bad);
        } else if (ok) {
            f = pmin(1.0, f * pmin(1.25, 0.97 * tolerance / pmax(H, 1e-12)));
        } else {
            f *= pmax(0.5, 0.95 * tolerance / H);
        }
    }
    if (have_ok) {
        out.res = std::move(best_ok_res);
        out.metrics = best_ok_m;
    } else {
        out.res = std::move(best_any_res);
        out.metrics = best_any_m;
    }
    out.within = have_ok;
    out.time_total = now() - t0;
    out.res.stats["auto"] = 1;
    out.res.stats["auto_tolerance"] = tolerance;
    out.res.stats["auto_within"] = have_ok;
    out.res.stats["time_total"] = out.time_total;
    return out;
}

SearchResult simplify_to_count(const PreparedMesh& prep, const BVH& bvh, i64 target_faces, const Options& options,
                               int max_passes, const PassProgress& progress) {
    double t0 = now();
    const VecD& P = prep.positions;
    const VecI& F = prep.faces;
    i64 n = target_faces;
    SearchResult out;
    SimplifyResult best = simplify(P, F, n, options);
    Metrics best_m = compare_meshes(P, F, best.positions, best.faces, 50000, &bvh);
    double H0 = best_m.hausdorff;
    out.passes.push_back({0.0, best.output_faces(), H0, true});
    if (progress) progress(0, best.output_faces());
    double tau = 0.6 * H0, lo = 0, hi = 0;
    bool has_lo = false, has_hi = false;
    for (int k = 0; k < max_passes; k++) {
        if (tau <= 0) break;
        Options o = options;
        o.local_tol = tau;
        o.max_error = tau;
        SimplifyResult res = simplify(P, F, n, o, &bvh);
        bool reached = res.output_faces() <= n;
        double Hm = std::nan("");
        i64 faces = res.output_faces();
        if (reached) {
            Metrics m = compare_meshes(P, F, res.positions, res.faces, 50000, &bvh);
            Hm = m.hausdorff;
            if (m.hausdorff < best_m.hausdorff) {
                best = std::move(res);
                best_m = m;
            }
            hi = has_hi ? pmin(hi, tau) : tau;
            has_hi = true;
        } else {
            lo = has_lo ? pmax(lo, tau) : tau;
            has_lo = true;
        }
        out.passes.push_back({tau, faces, Hm, reached});
        if (progress) progress(k + 1, faces);
        if (has_lo && has_hi) {
            if (hi / lo < 1.15) break;
            tau = std::sqrt(lo * hi);
        } else if (reached) {
            tau *= 0.5;
        } else {
            tau *= 1.6;
        }
    }
    out.res = std::move(best);
    out.metrics = best_m;
    out.plain_hausdorff = H0;
    out.time_total = now() - t0;
    out.res.stats["best_for_count"] = 1;
    out.res.stats["plain_hausdorff"] = H0;
    out.res.stats["time_total"] = out.time_total;
    return out;
}

}  // namespace faqem
