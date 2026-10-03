// Port of faqem/core.py. The Numba kernels are transliterated statement by statement: same
// expression trees, same evaluation order, same tie-breaking. Vectorised numpy code is rewritten
// as loops that reproduce numpy's rounding (see compat/npcompat.hpp).
#include "core.hpp"

#include <atomic>
#include <chrono>
#include <stdexcept>

#include "compat/fmath.hpp"
#include "parallel.hpp"
#include "prep.hpp"
#include "virtual_edges.hpp"

namespace faqem {

static const double INF = std::numeric_limits<double>::infinity();
static const int QN = 10;
enum { C_FACES, C_COLL, C_FLIP, C_LINK, C_DUP, C_STALE, C_REBUILD, C_VIRT, C_SINCE, C_DONE, C_ERR };
static const int N_COUNTERS = 16;

static inline double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------------------------
// options
// ---------------------------------------------------------------------------------------------
std::vector<std::string> Options::keys() {
    return {"w_area", "w_boundary", "w_normal", "w_plane_area", "virtual_edges", "virtual_tau",
            "min_edge_rel", "flip_threshold", "kappa_dimensionless", "boundary_window", "preserve_topology",
            "topology_penalty", "max_move_factor", "plane_area_clamp", "max_rebuilds", "max_error", "auto",
            "local_tol"};
}
bool Options::is_bool(const std::string& k) {
    return k == "virtual_edges" || k == "kappa_dimensionless" || k == "preserve_topology" || k == "auto";
}
bool Options::set(const std::string& k, double v) {
    if (k == "w_area") w_area = v;
    else if (k == "w_boundary") w_boundary = v;
    else if (k == "w_normal") w_normal = v;
    else if (k == "w_plane_area") w_plane_area = v;
    else if (k == "virtual_edges") virtual_edges = v != 0;
    else if (k == "virtual_tau") virtual_tau = v;
    else if (k == "min_edge_rel") min_edge_rel = v;
    else if (k == "flip_threshold") flip_threshold = v;
    else if (k == "kappa_dimensionless") kappa_dimensionless = v != 0;
    else if (k == "boundary_window") boundary_window = v;
    else if (k == "preserve_topology") preserve_topology = v != 0;
    else if (k == "topology_penalty") topology_penalty = v;
    else if (k == "max_move_factor") max_move_factor = v;
    else if (k == "plane_area_clamp") plane_area_clamp = v;
    else if (k == "max_rebuilds") max_rebuilds = v;
    else if (k == "max_error") max_error = v;
    else if (k == "auto") auto_mode = v != 0;
    else if (k == "local_tol") local_tol = v;
    else return false;
    return true;
}
double Options::get(const std::string& k) const {
    if (k == "w_area") return w_area;
    if (k == "w_boundary") return w_boundary;
    if (k == "w_normal") return w_normal;
    if (k == "w_plane_area") return w_plane_area;
    if (k == "virtual_edges") return virtual_edges;
    if (k == "virtual_tau") return virtual_tau;
    if (k == "min_edge_rel") return min_edge_rel;
    if (k == "flip_threshold") return flip_threshold;
    if (k == "kappa_dimensionless") return kappa_dimensionless;
    if (k == "boundary_window") return boundary_window;
    if (k == "preserve_topology") return preserve_topology;
    if (k == "topology_penalty") return topology_penalty;
    if (k == "max_move_factor") return max_move_factor;
    if (k == "plane_area_clamp") return plane_area_clamp;
    if (k == "max_rebuilds") return max_rebuilds;
    if (k == "max_error") return max_error;
    if (k == "auto") return auto_mode;
    if (k == "local_tol") return local_tol;
    throw std::invalid_argument("unknown option " + k);
}

Options preset(const std::string& name) {
    Options o;
    if (name == "Paper Table 1 (exact)") {
        o.w_area = 100.0;
        o.w_boundary = 500.0;
        o.w_normal = 0.01;
        o.w_plane_area = 1.0;
        o.virtual_edges = true;
    } else if (name == "Plain QEM (ablation baseline)") {
        o.w_area = 0.0;
        o.w_boundary = 0.0;
        o.w_normal = 0.0;
        o.w_plane_area = 0.0;
        o.virtual_edges = false;
    }
    return o;
}

// ---------------------------------------------------------------------------------------------
// quadric helpers
// ---------------------------------------------------------------------------------------------
// numpy array ** scalar (fast paths of numpy's power ufunc for float64)
static inline double array_power(double x, double e) {
    if (e == 2.0) return x * x;
    if (e == -1.0) return 1.0 / x;
    if (e == 0.5) return std::sqrt(x);
    if (e == 1.0) return x;
    if (e == 0.0) return 1.0;
    return np::py_pow(x, e);
}

// w * p p^T for p = [n, d]
static inline void plane_q(double a, double b, double c, double d, double w, double* out) {
    out[0] = w * (a * a);
    out[1] = w * (a * b);
    out[2] = w * (a * c);
    out[3] = w * (a * d);
    out[4] = w * (b * b);
    out[5] = w * (b * c);
    out[6] = w * (b * d);
    out[7] = w * (c * c);
    out[8] = w * (c * d);
    out[9] = w * (d * d);
}

// Q[:, k] += np.bincount(idx, weights=vals[:, k], minlength=len(Q)) for every k
static void scatter_add(VecD& Q, i64 nV, const i64* idx, i64 idx_stride, i64 n, const VecD& vals) {
    VecD bc(nV * QN);
    for (int k = 0; k < QN; k++) {
        std::fill(bc.begin(), bc.end(), 0.0);
        for (i64 i = 0; i < n; i++) bc[idx[i * idx_stride]] += vals[i * QN + k];
        for (i64 v = 0; v < nV; v++) Q[v * QN + k] = Q[v * QN + k] + bc[v];
    }
}

Quadrics build_quadrics(const VecD& P, const VecI& F, const VecI& edges, const VecI& edge_counts, const Options& o) {
    Quadrics R;
    i64 nV = (i64)P.size() / 3, nF = (i64)F.size() / 3;
    VecD& Q = R.Q;
    Q.assign(nV * QN, 0.0);
    VecD area(nF), nrm(nF * 3, 0.0), ln(nF);
    std::vector<char> ok(nF);
    VecD area_ok;
    for (i64 f = 0; f < nF; f++) {
        const double* a = &P[F[f * 3] * 3];
        const double* b = &P[F[f * 3 + 1] * 3];
        const double* c = &P[F[f * 3 + 2] * 3];
        double cx, cy, cz;
        np::cross(b[0] - a[0], b[1] - a[1], b[2] - a[2], c[0] - a[0], c[1] - a[1], c[2] - a[2], cx, cy, cz);
        ln[f] = np::norm3(cx, cy, cz);
        area[f] = ln[f] / 2;
        ok[f] = ln[f] > 1e-300;
        if (ok[f]) {
            nrm[f * 3] = cx / ln[f];
            nrm[f * 3 + 1] = cy / ln[f];
            nrm[f * 3 + 2] = cz / ln[f];
            area_ok.push_back(area[f]);
        }
    }
    double mean_area = area_ok.empty() ? 1.0 : np::mean(area_ok);

    // Q_base (Eq. 5)
    VecD w(nF, 1.0);
    if (o.w_plane_area != 0) {
        double cl = o.plane_area_clamp;
        for (i64 f = 0; f < nF; f++) {
            double rel = np::fmax0(area[f] / mean_area, 1e-6);
            w[f] = np::clip(array_power(rel, -o.w_plane_area), 1.0 / cl, cl);
        }
    }
    for (i64 f = 0; f < nF; f++)
        if (!ok[f]) w[f] = 0.0;
    VecD d(nF);
    for (i64 f = 0; f < nF; f++) {
        const double* a = &P[F[f * 3] * 3];
        d[f] = -np::einsum3(nrm[f * 3], nrm[f * 3 + 1], nrm[f * 3 + 2], a[0], a[1], a[2]);
    }
    VecD Kg(nF * QN), Kp(nF * QN);
    for (i64 f = 0; f < nF; f++) {
        plane_q(nrm[f * 3], nrm[f * 3 + 1], nrm[f * 3 + 2], d[f], ok[f] ? area[f] : 0.0, &Kg[f * QN]);
        plane_q(nrm[f * 3], nrm[f * 3 + 1], nrm[f * 3 + 2], d[f], w[f], &Kp[f * QN]);
    }
    R.Qgeo.assign(nV * QN, 0.0);
    R.Ageo.assign(nV, 0.0);
    for (int k = 0; k < 3; k++) {
        scatter_add(R.Qgeo, nV, &F[k], 3, nF, Kg);
        VecD bc(nV, 0.0);
        for (i64 f = 0; f < nF; f++) bc[F[f * 3 + k]] += area[f];
        for (i64 v = 0; v < nV; v++) R.Ageo[v] = R.Ageo[v] + bc[v];
    }
    for (int k = 0; k < 3; k++) scatter_add(Q, nV, &F[k], 3, nF, Kp);

    // Q_normal (Eq. 10)
    if (o.w_normal > 0) {
        VecD vn(nV * 3, 0.0);
        VecD ang(nF);
        for (int k = 0; k < 3; k++) {
            for (i64 f = 0; f < nF; f++) {
                const double* p0 = &P[F[f * 3 + k] * 3];
                const double* p1 = &P[F[f * 3 + (k + 1) % 3] * 3];
                const double* p2 = &P[F[f * 3 + (k + 2) % 3] * 3];
                double ux = p1[0] - p0[0], uy = p1[1] - p0[1], uz = p1[2] - p0[2];
                double vx = p2[0] - p0[0], vy = p2[1] - p0[1], vz = p2[2] - p0[2];
                double cx, cy, cz;
                np::cross(ux, uy, uz, vx, vy, vz, cx, cy, cz);
                ang[f] = fm::atan2(np::norm3(cx, cy, cz), np::einsum3(ux, uy, uz, vx, vy, vz));
            }
            for (int j = 0; j < 3; j++) {
                VecD bc(nV, 0.0);
                for (i64 f = 0; f < nF; f++) bc[F[f * 3 + k]] += ang[f] * nrm[f * 3 + j];
                for (i64 v = 0; v < nV; v++) vn[v * 3 + j] = vn[v * 3 + j] + bc[v];
            }
        }
        double qn[QN];
        for (i64 v = 0; v < nV; v++) {
            double l = np::norm3(vn[v * 3], vn[v * 3 + 1], vn[v * 3 + 2]);
            bool good = l > 1e-12;
            double un0 = 0, un1 = 0, un2 = 0;
            if (good) {
                un0 = vn[v * 3] / l;
                un1 = vn[v * 3 + 1] / l;
                un2 = vn[v * 3 + 2] / l;
            }
            double dn = -np::einsum3(un0, un1, un2, P[v * 3], P[v * 3 + 1], P[v * 3 + 2]);
            plane_q(un0, un1, un2, dn, good ? o.w_normal : 0.0, qn);
            for (int k = 0; k < QN; k++) Q[v * QN + k] = Q[v * QN + k] + qn[k];
        }
    }

    // Q_boundary (Eq. 6-9)
    i64 ne = (i64)edge_counts.size();
    VecD elen(ne);
    for (i64 e = 0; e < ne; e++) {
        const double* a = &P[edges[e * 2] * 3];
        const double* b = &P[edges[e * 2 + 1] * 3];
        elen[e] = np::norm3(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    }
    double mean_edge = ne ? np::mean(elen) : 1.0;
    VecI b0, b1;
    for (i64 e = 0; e < ne; e++)
        if (edge_counts[e] == 1) {
            b0.push_back(edges[e * 2]);
            b1.push_back(edges[e * 2 + 1]);
        }
    i64 nb = (i64)b0.size();
    i64 n_boundary_vertices = 0;
    if (nb) {
        VecI bv;
        for (i64 i = 0; i < nb; i++) {
            bv.push_back(b0[i]);
            bv.push_back(b1[i]);
        }
        std::sort(bv.begin(), bv.end());
        n_boundary_vertices = (i64)(std::unique(bv.begin(), bv.end()) - bv.begin());
    }
    if (o.w_boundary > 0 && nb) {
        VecI src(2 * nb), dst(2 * nb);
        for (i64 i = 0; i < nb; i++) {
            src[i] = b0[i];
            src[nb + i] = b1[i];
            dst[i] = b1[i];
            dst[nb + i] = b0[i];
        }
        VecI order = np::argsort_stable(src);
        VecI s2(2 * nb), d2(2 * nb);
        for (i64 i = 0; i < 2 * nb; i++) {
            s2[i] = src[order[i]];
            d2[i] = dst[order[i]];
        }
        VecI v1, n1_, n2_;
        for (i64 i = 0; i < 2 * nb;) {
            i64 j = i + 1;
            while (j < 2 * nb && s2[j] == s2[i]) j++;
            if (j - i == 2) {
                v1.push_back(s2[i]);
                n1_.push_back(d2[i]);
                n2_.push_back(d2[i + 1]);
            }
            i = j;
        }
        i64 nv1 = (i64)v1.size();
        VecI nb1(nV, -1), nb2(nV, -1);
        for (i64 i = 0; i < nv1; i++) {
            nb1[v1[i]] = n1_[i];
            nb2[v1[i]] = n2_[i];
        }
        i64 win = (i64)o.boundary_window;
        auto walk = [&](VecI prev, VecI cur, i64 steps) {
            for (i64 s = 0; s < steps - 1; s++) {
                for (i64 i = 0; i < nv1; i++) {
                    i64 c = cur[i];
                    bool okw = nb1[c] >= 0;
                    i64 nxt = nb1[c] == prev[i] ? nb2[c] : nb1[c];
                    nxt = (okw && nxt != v1[i]) ? nxt : c;
                    prev[i] = (nxt != c) ? c : prev[i];
                    cur[i] = nxt;
                }
            }
            return cur;
        };
        VecI v2 = walk(v1, n1_, win);
        VecI v3 = walk(v1, n2_, win);
        VecD Qb(nv1 * QN);
        double q1[QN], q2[QN];
        for (i64 i = 0; i < nv1; i++) {
            const double* x1 = &P[v1[i] * 3];
            const double* x2 = &P[v2[i] * 3];
            const double* x3 = &P[v3[i] * 3];
            double d1[3], dd2[3];
            for (int k = 0; k < 3; k++) {
                d1[k] = x3[k] - x2[k];
                dd2[k] = (x3[k] - 2 * x1[k]) + x2[k];
            }
            double dl = np::norm3(d1[0], d1[1], d1[2]);
            double cx, cy, cz;
            np::cross(d1[0], d1[1], d1[2], dd2[0], dd2[1], dd2[2], cx, cy, cz);
            double kappa = np::norm3(cx, cy, cz) / np::py_pow(np::fmax0(dl, 1e-12), 3.0);
            if (o.kappa_dimensionless) kappa = kappa * dl / 2;
            else kappa = np::fmin0(kappa, 2.0 / mean_edge);
            kappa = dl > 1e-12 ? kappa : 0.0;
            double wb = o.w_boundary * kappa;
            double n1[3];
            np::cross(x1[0] - x2[0], x1[1] - x2[1], x1[2] - x2[2], x3[0] - x1[0], x3[1] - x1[1], x3[2] - x1[2],
                      n1[0], n1[1], n1[2]);
            double l1 = np::norm3(n1[0], n1[1], n1[2]);
            bool g1 = l1 > 1e-14;
            double n1u[3] = {0, 0, 0};
            if (g1)
                for (int k = 0; k < 3; k++) n1u[k] = n1[k] / l1;
            double dd[3] = {x1[0] - x2[0], x1[1] - x2[1], x1[2] - x2[2]};
            double l2 = np::norm3(dd[0], dd[1], dd[2]);
            bool g2 = l2 > 1e-14;
            double du[3] = {0, 0, 0};
            if (g2)
                for (int k = 0; k < 3; k++) du[k] = dd[k] / l2;
            plane_q(n1u[0], n1u[1], n1u[2], -np::einsum3(n1u[0], n1u[1], n1u[2], x1[0], x1[1], x1[2]),
                    g1 ? wb : 0.0, q1);
            plane_q(du[0], du[1], du[2], -np::einsum3(du[0], du[1], du[2], x1[0], x1[1], x1[2]), g2 ? wb : 0.0,
                    q2);
            for (int k = 0; k < QN; k++) Qb[i * QN + k] = q1[k] + q2[k];
        }
        scatter_add(Q, nV, v1.data(), 1, nv1, Qb);
    }
    R.mean_area = mean_area;
    R.mean_edge = mean_edge;
    R.n_boundary_vertices = n_boundary_vertices;
    return R;
}

// =============================================================================================
// kernels (Numba ports)
// =============================================================================================
static inline double sq(double x) { return x * x; }

static inline double eval_q(const double* q, double x, double y, double z) {
    return (q[0] * x * x + 2 * q[1] * x * y + 2 * q[2] * x * z + 2 * q[3] * x + q[4] * y * y + 2 * q[5] * y * z +
            2 * q[6] * y + q[7] * z * z + 2 * q[8] * z + q[9]);
}

static void jacobi3(double* A, double* V) {
    for (int i = 0; i < 9; i++) V[i] = 0.0;
    V[0] = 1.0;
    V[4] = 1.0;
    V[8] = 1.0;
    for (int sweep = 0; sweep < 12; sweep++) {
        double off = A[1] * A[1] + A[2] * A[2] + A[5] * A[5];
        double dg = A[0] * A[0] + A[4] * A[4] + A[8] * A[8];
        if (off == 0.0 || off <= 1e-24 * dg) return;
        for (int p = 0; p < 2; p++) {
            for (int q = p + 1; q < 3; q++) {
                double apq = A[p * 3 + q];
                if (apq == 0.0) continue;
                double theta = (A[q * 3 + q] - A[p * 3 + p]) / (2 * apq);
                double sg = theta >= 0 ? 1.0 : -1.0;
                double t = sg / (std::fabs(theta) + std::sqrt(theta * theta + 1));
                double c = 1 / std::sqrt(t * t + 1);
                double s = t * c;
                for (int k = 0; k < 3; k++) {
                    double akp = A[k * 3 + p], akq = A[k * 3 + q];
                    A[k * 3 + p] = c * akp - s * akq;
                    A[k * 3 + q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; k++) {
                    double apk = A[p * 3 + k], aqk = A[q * 3 + k];
                    A[p * 3 + k] = c * apk - s * aqk;
                    A[q * 3 + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; k++) {
                    double vkp = V[k * 3 + p], vkq = V[k * 3 + q];
                    V[k * 3 + p] = c * vkp - s * vkq;
                    V[k * 3 + q] = s * vkp + c * vkq;
                }
            }
        }
    }
}

bool solve_q(const double* q, double mx, double my, double mz, double* out, double rel_eps, double* Am, double* Vm) {
    double a00 = q[0], a01 = q[1], a02 = q[2], a11 = q[4], a12 = q[5], a22 = q[7];
    double tr = a00 + a11 + a22;
    if (!(tr > 1e-300)) return false;
    double rx = a00 * mx + a01 * my + a02 * mz + q[3];
    double ry = a01 * mx + a11 * my + a12 * mz + q[6];
    double rz = a02 * mx + a12 * my + a22 * mz + q[8];
    double c00 = a11 * a22 - a12 * a12;
    double c01 = a02 * a12 - a01 * a22;
    double c02 = a01 * a12 - a02 * a11;
    double det = a00 * c00 + a01 * c01 + a02 * c02;
    if (det > rel_eps * 4 * tr * tr * tr) {
        double c11 = a00 * a22 - a02 * a02;
        double c12 = a01 * a02 - a00 * a12;
        double c22 = a00 * a11 - a01 * a01;
        double inv = 1.0 / det;
        out[0] = mx - (c00 * rx + c01 * ry + c02 * rz) * inv;
        out[1] = my - (c01 * rx + c11 * ry + c12 * rz) * inv;
        out[2] = mz - (c02 * rx + c12 * ry + c22 * rz) * inv;
        return true;
    }
    Am[0] = a00;
    Am[1] = a01;
    Am[2] = a02;
    Am[3] = a01;
    Am[4] = a11;
    Am[5] = a12;
    Am[6] = a02;
    Am[7] = a12;
    Am[8] = a22;
    jacobi3(Am, Vm);
    double lmax = std::max(std::max(Am[0], Am[4]), Am[8]);
    if (!(lmax > 0)) return false;
    double thr = rel_eps * lmax;
    double dx = 0.0, dy = 0.0, dz = 0.0;
    for (int k = 0; k < 3; k++) {
        double lk = Am[k * 4];
        if (lk <= thr) continue;
        double ux = Vm[k], uy = Vm[3 + k], uz = Vm[6 + k];
        double s = (ux * rx + uy * ry + uz * rz) / lk;
        dx -= s * ux;
        dy -= s * uy;
        dz -= s * uz;
    }
    out[0] = mx + dx;
    out[1] = my + dy;
    out[2] = mz + dz;
    return true;
}

namespace {

// pool-backed growable per-vertex lists
struct Lists {
    VecI pool, st, ln, cp;
    i64 used = 0;
    void push(i64 v, i64 val) {
        if (ln[v] == cp[v]) {
            i64 newcap = std::max<i64>(4, cp[v] * 2);
            if (used + newcap > (i64)pool.size()) pool.resize(std::max<i64>((i64)pool.size() * 2, used + newcap));
            i64 s = used;
            for (i64 i = 0; i < ln[v]; i++) pool[s + i] = pool[st[v] + i];
            st[v] = s;
            cp[v] = newcap;
            used += newcap;
        }
        pool[st[v] + ln[v]] = val;
        ln[v] += 1;
    }
    bool remove(i64 v, i64 val) {
        i64 s = st[v];
        for (i64 i = 0; i < ln[v]; i++) {
            if (pool[s + i] == val) {
                pool[s + i] = pool[s + ln[v] - 1];
                ln[v] -= 1;
                return true;
            }
        }
        return false;
    }
    bool has(i64 v, i64 val) const {
        i64 s = st[v];
        for (i64 i = 0; i < ln[v]; i++)
            if (pool[s + i] == val) return true;
        return false;
    }
};

// _build_lists: (src -> dst) pairs, stable by src
Lists build_lists(i64 nV, const VecI& lsrc, const VecI& ldst, i64 slack = 2) {
    Lists L;
    VecI order = np::argsort_stable(lsrc);
    i64 n = (i64)lsrc.size();
    VecI counts(nV, 0);
    for (i64 s : lsrc) counts[s]++;
    L.cp.resize(nV);
    for (i64 v = 0; v < nV; v++) L.cp[v] = std::max<i64>(counts[v] * slack, 4);
    L.st.assign(nV, 0);
    for (i64 v = 1; v < nV; v++) L.st[v] = L.st[v - 1] + L.cp[v - 1];
    i64 total = 0;
    for (i64 v = 0; v < nV; v++) total += L.cp[v];
    L.used = total;
    L.pool.assign((i64)((double)total * 1.25) + 16, 0);
    VecI first(nV, 0);
    for (i64 v = 1; v < nV; v++) first[v] = first[v - 1] + counts[v - 1];
    for (i64 i = 0; i < n; i++) {
        i64 s = lsrc[order[i]];
        L.pool[L.st[s] + (i - first[s])] = ldst[order[i]];
    }
    L.ln = counts;
    return L;
}

struct Heap {
    VecD H;  // rows [cost, a, b, stamp_a, stamp_b, x, y, z]
    i64 hs = 0;
    void push(double cost, double a, double b, double sa, double sb, double x, double y, double z) {
        i64 n = hs;
        if (n == (i64)(H.size() / 8)) H.resize(H.size() * 2);
        i64 i = n;
        hs = n + 1;
        while (i > 0) {
            i64 p = (i - 1) >> 1;
            if (H[p * 8] <= cost) break;
            for (int k = 0; k < 8; k++) H[i * 8 + k] = H[p * 8 + k];
            i = p;
        }
        double* r = &H[i * 8];
        r[0] = cost;
        r[1] = a;
        r[2] = b;
        r[3] = sa;
        r[4] = sb;
        r[5] = x;
        r[6] = y;
        r[7] = z;
    }
    void pop(double* out) {
        double tmp[8];
        for (int k = 0; k < 8; k++) out[k] = H[k];
        i64 n = hs - 1;
        hs = n;
        if (n == 0) return;
        for (int k = 0; k < 8; k++) tmp[k] = H[n * 8 + k];
        double cost = tmp[0];
        i64 i = 0;
        while (true) {
            i64 l = 2 * i + 1;
            if (l >= n) break;
            i64 r = l + 1;
            if (r < n && H[r * 8] < H[l * 8]) l = r;
            if (H[l * 8] >= cost) break;
            for (int k = 0; k < 8; k++) H[i * 8 + k] = H[l * 8 + k];
            i = l;
        }
        for (int k = 0; k < 8; k++) H[i * 8 + k] = tmp[k];
    }
};

// local worst-case guarantee data (the K tuple of core.py)
struct LH {
    VecD ptP, tol2;
    VecI fhead, pnext, pbest;
    VecD vtol2;
    VecI buf, fan;
    double T3[9] = {0};
    i64 F3[3] = {0, 1, 2};
    double tmp[6] = {0};
    i64 stack[256] = {0};
    double sd[256] = {0};
    double out[6] = {0};
    BVHView ref{};
    VecD fanT;  // fan triangle corners of the collapse being checked
    double fnrm[3] = {0};
    i64 reps[3] = {0};
};

struct Engine {
    VecD P, Q;
    VecI F;
    i64 nV = 0, nF = 0;
    Lists L, V;  // face lists, virtual neighbour lists
    Heap H;
    VecI mark, cnt, mark2;
    i64 tag = 0, tag2 = 0;
    VecI rbA, rbB;
    double prm[10];
    i64 C[N_COUNTERS] = {0};
    VecD Qgeo, Ageo;
    LH K;
    std::vector<uint8_t> face_alive, v_alive;
    VecI stamp, parent;
    VecD hist;
    // run_collapses scratch
    VecI cA, oppc;

    inline bool face_has(i64 f, i64 v) const { return F[f * 3] == v || F[f * 3 + 1] == v || F[f * 3 + 2] == v; }

    i64 ring(i64 v, i64* out) {
        tag += 1;
        i64 t = tag;
        i64 n = 0;
        i64 s = L.st[v];
        for (i64 i = 0; i < L.ln[v]; i++) {
            i64 f = L.pool[s + i];
            for (int k = 0; k < 3; k++) {
                i64 u = F[f * 3 + k];
                if (u == v) continue;
                if (mark[u] != t) {
                    mark[u] = t;
                    cnt[u] = 0;
                    out[n] = u;
                    n += 1;
                }
                cnt[u] += 1;
            }
        }
        return n;
    }

    void add_area_edge(double* QA, i64 r, i64 s, double h) {
        double rx = P[r * 3], ry = P[r * 3 + 1], rz = P[r * 3 + 2];
        double sx = P[s * 3], sy = P[s * 3 + 1], sz = P[s * 3 + 2];
        double ex = sx - rx, ey = sy - ry, ez = sz - rz;
        double tx = ry * sz - rz * sy, ty = rz * sx - rx * sz, tz = rx * sy - ry * sx;
        double ee = ex * ex + ey * ey + ez * ez;
        QA[0] += h * (ee - ex * ex);
        QA[1] += h * (-ex * ey);
        QA[2] += h * (-ex * ez);
        QA[4] += h * (ee - ey * ey);
        QA[5] += h * (-ey * ez);
        QA[7] += h * (ee - ez * ez);
        QA[3] += h * (ty * ez - tz * ey);
        QA[6] += h * (tz * ex - tx * ez);
        QA[8] += h * (tx * ey - ty * ex);
        QA[9] += h * (tx * tx + ty * ty + tz * tz);
    }

    inline double seg_d2(double px, double py, double pz, i64 i, i64 j, double x2, double y2, double z2,
                         bool use_xyz) const {
        double ax = P[i * 3], ay = P[i * 3 + 1], az = P[i * 3 + 2];
        double bx, by, bz;
        if (use_xyz) {
            bx = x2;
            by = y2;
            bz = z2;
        } else {
            bx = P[j * 3];
            by = P[j * 3 + 1];
            bz = P[j * 3 + 2];
        }
        double ex = bx - ax, ey = by - ay, ez = bz - az;
        double ll = ex * ex + ey * ey + ez * ez;
        double qx = px - ax, qy = py - ay, qz = pz - az;
        double t = ll > 0 ? (qx * ex + qy * ey + qz * ez) / ll : 0.0;
        t = std::min(1.0, std::max(0.0, t));
        double dx = qx - t * ex, dy = qy - t * ey, dz = qz - t * ez;
        return dx * dx + dy * dy + dz * dz;
    }

    double boundary_dev2(i64 a, i64 b, double x, double y, double z) {
        i64* rbA_ = rbA.data();
        i64* rbB_ = rbB.data();
        i64 na = ring(a, rbA_);
        bool ab_bnd = false;
        i64 nA = 0;
        for (i64 i = 0; i < na; i++) {
            i64 u = rbA_[i];
            if (cnt[u] == 1) {
                if (u == b) ab_bnd = true;
                else {
                    rbA_[nA] = u;
                    nA += 1;
                }
            }
        }
        i64 surv = 0;
        for (i64 i = 0; i < L.ln[a]; i++)
            if (!face_has(L.pool[L.st[a] + i], b)) surv += 1;
        for (i64 i = 0; i < L.ln[b]; i++)
            if (!face_has(L.pool[L.st[b] + i], a)) surv += 1;
        if (surv == 0) return INF;
        i64 nb = ring(b, rbB_);
        i64 nB = 0;
        for (i64 i = 0; i < nb; i++) {
            i64 u = rbB_[i];
            if (cnt[u] == 1 && u != a) {
                rbB_[nB] = u;
                nB += 1;
            }
        }
        if (nA == 0 && nB == 0 && !ab_bnd) return -1.0;
        double worst = 0.0;
        for (int pi = 0; pi < 2; pi++) {
            i64 p = pi == 0 ? a : b;
            bool on_b = ab_bnd || (pi == 0 ? nA > 0 : nB > 0);
            if (!on_b) continue;
            double dmin = INF;
            for (i64 i = 0; i < nA + nB; i++) {
                i64 n = i < nA ? rbA_[i] : rbB_[i - nA];
                dmin = std::min(dmin, seg_d2(P[p * 3], P[p * 3 + 1], P[p * 3 + 2], n, n, x, y, z, true));
            }
            worst = std::max(worst, dmin);
        }
        double dmin = INF;
        if (ab_bnd) dmin = seg_d2(x, y, z, a, b, 0.0, 0.0, 0.0, false);
        for (i64 i = 0; i < nA; i++) dmin = std::min(dmin, seg_d2(x, y, z, a, rbA_[i], 0.0, 0.0, 0.0, false));
        for (i64 i = 0; i < nB; i++) dmin = std::min(dmin, seg_d2(x, y, z, b, rbB_[i], 0.0, 0.0, 0.0, false));
        return std::max(worst, dmin);
    }

    // COMPUTECOST: writes v' into best, returns cost_total
    double eval_edge(i64 a, i64 b, double* S, double* best) {
        double w_area = prm[0], max_move = prm[2], min_edge2 = prm[3], area_norm = prm[4], rel_eps = prm[6];
        double ax = P[a * 3], ay = P[a * 3 + 1], az = P[a * 3 + 2];
        double bx = P[b * 3], by = P[b * 3 + 1], bz = P[b * 3 + 2];
        double len2 = sq(ax - bx) + sq(ay - by) + sq(az - bz);
        if (len2 < min_edge2) return INF;
        double* Qs = S;
        double* QA = S + 10;
        double* sol = S + 20;
        double* Am = S + 23;
        double* Vm = S + 32;
        for (int k = 0; k < QN; k++) Qs[k] = Q[a * QN + k] + Q[b * QN + k];
        bool has_area = false;
        if (w_area > 0) {
            for (int k = 0; k < QN; k++) QA[k] = 0.0;
            double h = 0.5 * area_norm;
            i64 n = ring(a, rbA.data());
            for (i64 i = 0; i < n; i++) {
                if (cnt[rbA[i]] == 1) {
                    add_area_edge(QA, a, rbA[i], h);
                    has_area = true;
                }
            }
            n = ring(b, rbB.data());
            for (i64 i = 0; i < n; i++) {
                i64 u = rbB[i];
                if (cnt[u] == 1 && u != a) {
                    add_area_edge(QA, b, u, h);
                    has_area = true;
                }
            }
        }
        double mx = (ax + bx) / 2, my = (ay + by) / 2, mz = (az + bz) / 2;
        double best_cost = INF;
        for (int cand = 0; cand < 4; cand++) {
            S[41 + cand] = INF;
            double x, y, z;
            if (cand == 0) {
                if (!solve_q(Qs, mx, my, mz, sol, rel_eps, Am, Vm)) continue;
                x = sol[0];
                y = sol[1];
                z = sol[2];
                if (sq(x - mx) + sq(y - my) + sq(z - mz) > max_move * max_move * len2) continue;
            } else if (cand == 1) {
                x = ax;
                y = ay;
                z = az;
            } else if (cand == 2) {
                x = bx;
                y = by;
                z = bz;
            } else {
                x = mx;
                y = my;
                z = mz;
            }
            double c = eval_q(Qs, x, y, z);
            if (has_area) c += w_area * eval_q(QA, x, y, z);
            S[41 + cand] = std::max(0.0, c);
            S[45 + 3 * cand] = x;
            S[46 + 3 * cand] = y;
            S[47 + 3 * cand] = z;
            if (c < best_cost) {
                best_cost = c;
                best[0] = x;
                best[1] = y;
                best[2] = z;
            }
        }
        return std::max(0.0, best_cost);
    }

    bool flip_ok(i64 v, i64 other, double x, double y, double z, double thr, double degthr = 1e-40) const {
        i64 s = L.st[v];
        for (i64 i = 0; i < L.ln[v]; i++) {
            i64 f = L.pool[s + i];
            if (face_has(f, other)) continue;
            i64 i0 = F[f * 3], i1 = F[f * 3 + 1], i2 = F[f * 3 + 2];
            double ux = P[i1 * 3] - P[i0 * 3], uy = P[i1 * 3 + 1] - P[i0 * 3 + 1], uz = P[i1 * 3 + 2] - P[i0 * 3 + 2];
            double wx = P[i2 * 3] - P[i0 * 3], wy = P[i2 * 3 + 1] - P[i0 * 3 + 1], wz = P[i2 * 3 + 2] - P[i0 * 3 + 2];
            double ox = uy * wz - uz * wy, oy = uz * wx - ux * wz, oz = ux * wy - uy * wx;
            double q0x = i0 == v ? x : P[i0 * 3], q0y = i0 == v ? y : P[i0 * 3 + 1], q0z = i0 == v ? z : P[i0 * 3 + 2];
            double q1x = i1 == v ? x : P[i1 * 3], q1y = i1 == v ? y : P[i1 * 3 + 1], q1z = i1 == v ? z : P[i1 * 3 + 2];
            double q2x = i2 == v ? x : P[i2 * 3], q2y = i2 == v ? y : P[i2 * 3 + 1], q2z = i2 == v ? z : P[i2 * 3 + 2];
            ux = q1x - q0x;
            uy = q1y - q0y;
            uz = q1z - q0z;
            wx = q2x - q0x;
            wy = q2y - q0y;
            wz = q2z - q0z;
            double nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
            double oo = ox * ox + oy * oy + oz * oz;
            double nn = nx * nx + ny * ny + nz * nz;
            if (oo < degthr) continue;
            if (nn < 1e-12 * oo) return false;
            if (ox * nx + oy * ny + oz * nz < thr * std::sqrt(oo * nn)) return false;
        }
        return true;
    }

    inline i64 ref_closest(double x, double y, double z, double max_d2, double& d2) {
        return bvh_closest(K.ref, x, y, z, -1, 0, K.reps, -1, -1, -1, K.fnrm, 0.0, 0.0, 0.0, K.stack, K.sd, 256,
                           K.out, K.tmp, max_d2, d2);
    }

    // Phases 2 and 3 run their independent tests on all cores when the work is large; each phase
    // is an AND over its tests, so the outcome is identical for any number of threads (see
    // parallel.hpp). pbest writes of a failed check are never read (lh_attach runs only after a
    // successful check, which rewrites every pbest it covers).
    // below these sizes a check runs on one core: handing it to the pool would cost more than it saves
    static constexpr i64 PAR_MIN_TESTS = 4096;   // point x triangle tests (phase 3)
    static constexpr i64 PAR_MIN_SAMPLES = 64;   // BVH queries (phase 2)
    int nthreads = 1;                            // set once per simplify()

    struct Scratch {
        i64 stack[256];
        double sd[256], out[6], tmp[6];
    };
    std::vector<Scratch> scratch;
    VecD samples;  // phase 2 sample points
    VecI pts;      // phase 3 tracked point ids

    inline i64 ref_closest_s(Scratch& sc, double x, double y, double z, double max_d2, double& d2) {
        return bvh_closest(K.ref, x, y, z, -1, 0, K.reps, -1, -1, -1, K.fnrm, 0.0, 0.0, 0.0, sc.stack, sc.sd, 256,
                           sc.out, sc.tmp, max_d2, d2);
    }

    // best fan face of tracked point p (same loop and early break as the sequential code)
    inline bool fit_point(i64 p, i64 nf, const i64* fan, double* tmp) {
        double px = K.ptP[p * 3], py = K.ptP[p * 3 + 1], pz = K.ptP[p * 3 + 2];
        double t = K.tol2[p];
        double best = INF;
        i64 bf = -1;
        for (i64 j = 0; j < nf; j++) {
            double dd = closest_on_tri(&K.fanT[j * 9], K.F3, 0, px, py, pz, tmp);
            if (dd < best) {
                best = dd;
                bf = fan[j];
                if (dd <= 0.01 * t) break;
            }
        }
        if (best > t) return false;
        K.pbest[p] = bf;
        return true;
    }

    bool lh_check(i64 a, i64 b, double x, double y, double z) {
        i64* fan = K.fan.data();
        i64 fan_len = (i64)K.fan.size();
        double* T3 = K.T3;
        i64 nf = 0;
        for (i64 i = 0; i < L.ln[a]; i++) {
            i64 f = L.pool[L.st[a] + i];
            if (!face_has(f, b) && nf < fan_len) {
                fan[nf] = f;
                nf += 1;
            }
        }
        for (i64 i = 0; i < L.ln[b]; i++) {
            i64 f = L.pool[L.st[b] + i];
            if (!face_has(f, a) && nf < fan_len) {
                fan[nf] = f;
                nf += 1;
            }
        }
        if (nf == 0) return true;
        bool par = nthreads > 1;
        double lim = std::min(K.vtol2[a], K.vtol2[b]);
        double d2;
        // phase 1: the new vertex vs the original surface
        i64 fb = ref_closest(x, y, z, lim * 1.0001, d2);
        if (fb < 0 || d2 > lim) return false;
        // phase 2: points sampled on the new faces vs the original surface
        double sl = std::sqrt(lim);
        samples.clear();
        for (i64 j = 0; j < nf; j++) {
            i64 g = fan[j];
            int kv = 0;
            for (int k = 0; k < 3; k++) {
                i64 u = F[g * 3 + k];
                if (u == a || u == b) {
                    T3[k * 3] = x;
                    T3[k * 3 + 1] = y;
                    T3[k * 3 + 2] = z;
                    kv = k;
                } else {
                    T3[k * 3] = P[u * 3];
                    T3[k * 3 + 1] = P[u * 3 + 1];
                    T3[k * 3 + 2] = P[u * 3 + 2];
                }
            }
            double emax = 0.0;
            for (int k = 0; k < 3; k++) {
                int k2 = (k + 1) % 3;
                double e = std::sqrt(sq(T3[k * 3] - T3[k2 * 3]) + sq(T3[k * 3 + 1] - T3[k2 * 3 + 1]) +
                                     sq(T3[k * 3 + 2] - T3[k2 * 3 + 2]));
                emax = std::max(emax, e);
            }
            i64 ng = (i64)std::min(8.0, std::max(2.0, std::ceil(emax / (3.0 * sl))));
            for (i64 gi = 0; gi < ng + 1; gi++) {
                for (i64 gj = 0; gj < ng + 1 - gi; gj++) {
                    i64 gk = ng - gi - gj;
                    if (gi == ng || gj == ng || gk == ng) continue;
                    if ((kv == 0 && gi == 0) || (kv == 1 && gj == 0) || (kv == 2 && gk == 0)) continue;
                    double wa = (double)gi / (double)ng, wb = (double)gj / (double)ng;
                    double wc = 1.0 - wa - wb;
                    double qx = wa * T3[0] + wb * T3[3] + wc * T3[6];
                    double qy = wa * T3[1] + wb * T3[4] + wc * T3[7];
                    double qz = wa * T3[2] + wb * T3[5] + wc * T3[8];
                    if (!par) {
                        fb = ref_closest(qx, qy, qz, lim * 1.0001, d2);
                        if (fb < 0 || d2 > lim) return false;
                    } else {
                        samples.push_back(qx);
                        samples.push_back(qy);
                        samples.push_back(qz);
                    }
                }
            }
        }
        if (par && (i64)samples.size() / 3 < PAR_MIN_SAMPLES) {
            for (size_t i = 0; i < samples.size(); i += 3) {
                fb = ref_closest(samples[i], samples[i + 1], samples[i + 2], lim * 1.0001, d2);
                if (fb < 0 || d2 > lim) return false;
            }
        } else if (par) {
            i64 ns = (i64)samples.size() / 3;
            std::atomic<bool> fail(false);
            parallel_for(ns, [&](i64 s, i64 e, int tid) {
                Scratch& sc = scratch[tid];
                for (i64 i = s; i < e && !fail.load(std::memory_order_relaxed); i++) {
                    double dd;
                    i64 f2 = ref_closest_s(sc, samples[i * 3], samples[i * 3 + 1], samples[i * 3 + 2], lim * 1.0001, dd);
                    if (f2 < 0 || dd > lim) fail.store(true, std::memory_order_relaxed);
                }
            }, std::max<i64>(4, ns / (4 * nthreads)));
            if (fail.load()) return false;
        }
        // the fan triangles after the collapse (built once; the Python rebuilds the same values per point)
        if ((i64)K.fanT.size() < nf * 9) K.fanT.resize(nf * 9);
        for (i64 j = 0; j < nf; j++) {
            i64 g = fan[j];
            double* t9 = &K.fanT[j * 9];
            for (int k = 0; k < 3; k++) {
                i64 u = F[g * 3 + k];
                if (u == a || u == b) {
                    t9[k * 3] = x;
                    t9[k * 3 + 1] = y;
                    t9[k * 3 + 2] = z;
                } else {
                    t9[k * 3] = P[u * 3];
                    t9[k * 3 + 1] = P[u * 3 + 1];
                    t9[k * 3 + 2] = P[u * 3 + 2];
                }
            }
        }
        // phase 3: every tracked point around a and b vs the new fan
        pts.clear();
        for (int side = 0; side < 2; side++) {
            i64 v = side == 0 ? a : b;
            i64 o = side == 0 ? b : a;
            for (i64 i = 0; i < L.ln[v]; i++) {
                i64 f = L.pool[L.st[v] + i];
                if (side == 1 && face_has(f, o)) continue;
                for (i64 p = K.fhead[f]; p >= 0; p = K.pnext[p]) pts.push_back(p);
            }
        }
        i64 np_ = (i64)pts.size();
        if (!par || np_ * nf < PAR_MIN_TESTS) {
            for (i64 i = 0; i < np_; i++)
                if (!fit_point(pts[i], nf, fan, K.tmp)) return false;
            return true;
        }
        std::atomic<bool> fail(false);
        parallel_for(np_, [&](i64 s, i64 e, int tid) {
            double* tmp = scratch[tid].tmp;
            for (i64 i = s; i < e && !fail.load(std::memory_order_relaxed); i++)
                if (!fit_point(pts[i], nf, fan, tmp)) fail.store(true, std::memory_order_relaxed);
        }, std::max<i64>(8, np_ / (4 * nthreads)));
        return !fail.load();
    }

    i64 lh_relink(i64 a, i64 b) {
        i64 n = 0;
        for (int side = 0; side < 2; side++) {
            i64 v = side == 0 ? a : b;
            i64 o = side == 0 ? b : a;
            for (i64 i = 0; i < L.ln[v]; i++) {
                i64 f = L.pool[L.st[v] + i];
                if (side == 1 && face_has(f, o)) continue;
                i64 p = K.fhead[f];
                while (p >= 0) {
                    K.buf[n] = p;
                    n += 1;
                    p = K.pnext[p];
                }
                K.fhead[f] = -1;
            }
        }
        return n;
    }

    void lh_attach(i64 n) {
        for (i64 i = 0; i < n; i++) {
            i64 p = K.buf[i];
            i64 g = K.pbest[p];
            K.pnext[p] = K.fhead[g];
            K.fhead[g] = p;
        }
    }

    // 0 reject, 1 real OK, 2 virtual OK, 3 flip only, 4 would change topology
    int can_collapse(i64 a, i64 b, double x, double y, double z, bool allow_topo) {
        i64* rbA_ = rbA.data();
        i64* rbB_ = rbB.data();
        if (prm[8] < INF) {
            double cap = prm[8];
            double qg[QN];
            for (int k = 0; k < QN; k++) qg[k] = Qgeo[a * QN + k] + Qgeo[b * QN + k];
            double e = eval_q(qg, x, y, z) / std::max(Ageo[a] + Ageo[b], 1e-300);
            if (e > cap) {
                C[C_ERR] += 1;
                return 0;
            }
            double bd = boundary_dev2(a, b, x, y, z);
            if (bd > cap) {
                C[C_ERR] += 1;
                return 0;
            }
        }
        tag2 += 1;
        i64 t2 = tag2;
        i64 shared = 0;
        i64 sb = L.st[b];
        for (i64 i = 0; i < L.ln[b]; i++) {
            i64 f = L.pool[sb + i];
            if (!face_has(f, a)) continue;
            shared += 1;
            for (int k = 0; k < 3; k++) {
                i64 u = F[f * 3 + k];
                if (u != a && u != b) {
                    if (mark2[u] != t2) {
                        mark2[u] = t2;
                        oppc[u] = 0;
                    }
                    oppc[u] += 1;
                }
            }
        }
        bool virt = shared == 0;
        if (virt) {
            if (!V.has(a, b)) return 0;
        } else if (prm[5] > 0 && !allow_topo) {
            i64 n = ring(a, rbA_);
            i64 ta = tag;
            bool a_bound = false;
            for (i64 i = 0; i < n; i++) {
                if (cnt[rbA_[i]] == 1) {
                    a_bound = true;
                    break;
                }
            }
            for (i64 i = 0; i < L.ln[b]; i++) {
                i64 f = L.pool[sb + i];
                for (int k = 0; k < 3; k++) {
                    i64 u = F[f * 3 + k];
                    if (u == a || u == b) continue;
                    if (mark[u] == ta && mark2[u] != t2) {
                        C[C_LINK] += 1;
                        return 4;
                    }
                }
            }
            if (shared >= 2 && a_bound) {
                n = ring(b, rbB_);
                for (i64 i = 0; i < n; i++) {
                    if (cnt[rbB_[i]] == 1) {
                        C[C_LINK] += 1;
                        return 4;
                    }
                }
            }
        }
        i64 n = ring(a, rbA_);
        for (i64 i = 0; i < n; i++) cA[rbA_[i]] = cnt[rbA_[i]];
        i64 nb = ring(b, rbB_);
        bool bad = false;
        for (i64 i = 0; i < nb; i++) {
            i64 u = rbB_[i];
            i64 ca = cA[u];
            if (u == a || ca == 0) continue;
            i64 cb = cnt[u];
            i64 o_ = mark2[u] == t2 ? oppc[u] : 0;
            i64 nw = ca + cb - 2 * o_;
            if (nw > 2 && nw > std::max(ca, cb)) {
                bad = true;
                break;
            }
        }
        for (i64 i = 0; i < n; i++) cA[rbA_[i]] = 0;
        if (bad && !allow_topo) {
            C[C_LINK] += 1;
            return 4;
        }
        i64 sa = L.st[a];
        for (i64 i = 0; i < L.ln[b]; i++) {
            i64 f = L.pool[sb + i];
            if (face_has(f, a)) continue;
            i64 x1 = -1, y1 = -1;
            for (int k = 0; k < 3; k++) {
                i64 u = F[f * 3 + k];
                if (u != b) {
                    if (x1 < 0) x1 = u;
                    else y1 = u;
                }
            }
            if (allow_topo) break;
            for (i64 j = 0; j < L.ln[a]; j++) {
                i64 g = L.pool[sa + j];
                if (face_has(g, x1) && face_has(g, y1)) {
                    C[C_DUP] += 1;
                    return 4;
                }
            }
        }
        if (!flip_ok(a, b, x, y, z, prm[1]) || !flip_ok(b, a, x, y, z, prm[1])) {
            C[C_FLIP] += 1;
            return 3;
        }
        if (prm[9] > 0 && !lh_check(a, b, x, y, z)) {
            C[C_ERR] += 1;
            return 0;
        }
        return virt ? 2 : 1;
    }

    void push_edge(i64 a, i64 b, double* S, double* best) {
        double c = eval_edge(a, b, S, best);
        if (c < INF) H.push(c, (double)a, (double)b, (double)stamp[a], (double)stamp[b], best[0], best[1], best[2]);
    }

    void run_collapses(i64 target_faces, i64 stop_faces, i64 max_rebuilds) {
        double S[64] = {0};
        double best[3] = {0};
        double top[8] = {0};
        i64 stop = std::max(target_faces, stop_faces);
        while (C[C_FACES] > stop) {
            if (H.hs == 0) {
                if (C[C_SINCE] == 0 || C[C_REBUILD] >= max_rebuilds) {
                    C[C_DONE] = 1;
                    break;
                }
                C[C_REBUILD] += 1;
                C[C_SINCE] = 0;
                for (i64 a = 0; a < nV; a++) {
                    if (!v_alive[a]) continue;
                    i64 n = ring(a, rbA.data());
                    for (i64 i = 0; i < n; i++) {
                        i64 u = rbA[i];
                        rbB[i] = u > a ? u : -1;
                    }
                    VecI nb(rbB.begin(), rbB.begin() + n);
                    for (i64 u : nb)
                        if (u >= 0) push_edge(a, u, S, best);
                    for (i64 i = 0; i < V.ln[a]; i++) {
                        i64 u = V.pool[V.st[a] + i];
                        if (u > a) push_edge(a, u, S, best);
                    }
                }
                if (H.hs == 0) {
                    C[C_DONE] = 1;
                    break;
                }
            }
            H.pop(top);
            i64 a = (i64)top[1], b = (i64)top[2];
            if (!v_alive[a] || !v_alive[b] || stamp[a] != (i64)top[3] || stamp[b] != (i64)top[4]) {
                C[C_STALE] += 1;
                continue;
            }
            double x = top[5], y = top[6], z = top[7];
            bool topo_ok = top[3] - std::floor(top[3]) > 0.25;
            int ok = can_collapse(a, b, x, y, z, topo_ok);
            if (ok == 4) {
                if (prm[7] < INF)
                    H.push(top[0] * prm[7] + 1e-12, (double)a, (double)b, (double)stamp[a] + 0.5, (double)stamp[b], x,
                           y, z);
                continue;
            }
            if (ok == 3) {
                eval_edge(a, b, S, best);
                for (int r = 0; r < 4; r++) {
                    int bi = -1;
                    double bc = INF;
                    for (int c = 0; c < 4; c++) {
                        if (S[41 + c] < bc) {
                            bc = S[41 + c];
                            bi = c;
                        }
                    }
                    if (bi < 0) break;
                    S[41 + bi] = INF;
                    double cx = S[45 + 3 * bi], cy = S[46 + 3 * bi], cz = S[47 + 3 * bi];
                    if (cx == x && cy == y && cz == z) continue;
                    if (flip_ok(a, b, cx, cy, cz, prm[1]) && flip_ok(b, a, cx, cy, cz, prm[1])) {
                        H.push(std::max(bc, top[0]), (double)a, (double)b, (double)stamp[a] + (topo_ok ? 0.5 : 0.0),
                               (double)stamp[b], cx, cy, cz);
                        break;
                    }
                }
                continue;
            }
            if (ok == 0) continue;
            if (ok == 2) C[C_VIRT] += 1;
            // COLLAPSEEDGE
            bool at_a = x == P[a * 3] && y == P[a * 3 + 1] && z == P[a * 3 + 2];
            bool at_b = x == P[b * 3] && y == P[b * 3 + 1] && z == P[b * 3 + 2];
            if (at_b && !at_a) std::swap(a, b);
            else if (!at_a && L.ln[b] > L.ln[a]) std::swap(a, b);
            i64 hk = C[C_COLL];
            if (hk < nV) {
                double* hr = &hist[hk * 7];
                hr[0] = (double)a;
                hr[1] = (double)b;
                hr[2] = x;
                hr[3] = y;
                hr[4] = z;
                hr[5] = top[0];
                hr[6] = (double)ok;
            }
            i64 lh_n = 0;
            if (prm[9] > 0) {
                lh_n = lh_relink(a, b);
                K.vtol2[a] = std::min(K.vtol2[a], K.vtol2[b]);
            }
            P[a * 3] = x;
            P[a * 3 + 1] = y;
            P[a * 3 + 2] = z;
            for (int k = 0; k < QN; k++) {
                Q[a * QN + k] += Q[b * QN + k];
                Qgeo[a * QN + k] += Qgeo[b * QN + k];
            }
            Ageo[a] += Ageo[b];
            bool removed = false;
            i64 sbb = L.st[b];
            for (i64 i = 0; i < L.ln[b]; i++) {
                i64 f = L.pool[sbb + i];
                if (face_has(f, a)) {
                    face_alive[f] = 0;
                    C[C_FACES] -= 1;
                    removed = true;
                    for (int k = 0; k < 3; k++) {
                        i64 u = F[f * 3 + k];
                        if (u != a && u != b) L.remove(u, f);
                    }
                } else {
                    for (int k = 0; k < 3; k++)
                        if (F[f * 3 + k] == b) F[f * 3 + k] = a;
                    L.push(a, f);
                    sbb = L.st[b];  // (push may move a's list only; b's start is unchanged)
                }
            }
            if (removed) {
                i64 s = L.st[a];
                i64 w = 0;
                for (i64 i = 0; i < L.ln[a]; i++) {
                    i64 f = L.pool[s + i];
                    if (face_alive[f]) {
                        L.pool[s + w] = f;
                        w += 1;
                    }
                }
                L.ln[a] = w;
            }
            if (topo_ok) {
                i64 s = L.st[a];
                i64 i = 0;
                while (i < L.ln[a]) {
                    i64 f = L.pool[s + i];
                    i64 dup = -1;
                    for (i64 j = i + 1; j < L.ln[a]; j++) {
                        i64 g = L.pool[s + j];
                        if (face_has(g, F[f * 3]) && face_has(g, F[f * 3 + 1]) && face_has(g, F[f * 3 + 2])) {
                            dup = g;
                            break;
                        }
                    }
                    if (dup < 0) {
                        i += 1;
                        continue;
                    }
                    bool same = false;
                    for (int k = 0; k < 3; k++)
                        if (F[f * 3] == F[dup * 3 + k] && F[f * 3 + 1] == F[dup * 3 + (k + 1) % 3]) same = true;
                    int kill_n = same ? 1 : 2;
                    for (int kk = 0; kk < kill_n; kk++) {
                        i64 g = kk == 0 ? dup : f;
                        face_alive[g] = 0;
                        C[C_FACES] -= 1;
                        for (int k = 0; k < 3; k++) L.remove(F[g * 3 + k], g);
                    }
                    i = 0;
                }
            }
            L.ln[b] = 0;
            v_alive[b] = 0;
            parent[b] = a;
            if (prm[9] > 0) lh_attach(lh_n);
            stamp[a] += 1;
            stamp[b] += 1;
            for (i64 i = 0; i < V.ln[b]; i++) {
                i64 u = V.pool[V.st[b] + i];
                V.remove(u, b);
                if (u == a) continue;
                if (!V.has(u, a)) {
                    V.push(u, a);
                    V.push(a, u);
                }
            }
            V.ln[b] = 0;
            // UPDATENEIGHBORCOSTS
            i64 n = ring(a, rbA.data());
            VecI nb(rbA.begin(), rbA.begin() + n);
            for (i64 u : nb) push_edge(a, u, S, best);
            for (i64 i = 0; i < V.ln[a]; i++) {
                i64 u = V.pool[V.st[a] + i];
                push_edge(a, u, S, best);
            }
            C[C_COLL] += 1;
            C[C_SINCE] += 1;
        }
    }
};

// _nearest_incident_face
void nearest_incident_face(const VecD& pts, const VecI& v0, const Engine& E, VecI& out) {
    double T3[9];
    i64 F3[3] = {0, 1, 2};
    double tmp[6];
    i64 n = (i64)v0.size();
    out.assign(n, -1);
    for (i64 i = 0; i < n; i++) {
        i64 v = v0[i];
        double best = INF;
        i64 bf = -1;
        for (i64 j = 0; j < E.L.ln[v]; j++) {
            i64 f = E.L.pool[E.L.st[v] + j];
            for (int k = 0; k < 3; k++)
                for (int c = 0; c < 3; c++) T3[k * 3 + c] = E.P[E.F[f * 3 + k] * 3 + c];
            double d = closest_on_tri(T3, F3, 0, pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2], tmp);
            if (d < best) {
                best = d;
                bf = f;
            }
        }
        out[i] = bf;
    }
}

}  // namespace

SimplifyResult simplify(const VecD& P_in, const VecI& F_in, i64 target_faces, const Options& options,
                        const BVH* reference) {
    Options o = options;
    if (o.auto_mode) {
        if (!o.max_error) throw std::invalid_argument("auto mode needs max_error (the allowed deviation)");
        target_faces = 1;
        o.topology_penalty = INF;
    }
    double t0 = now();
    Engine E;
    E.P = P_in;
    E.F = F_in;
    i64 nV = (i64)E.P.size() / 3, nF = (i64)E.F.size() / 3;
    E.nV = nV;
    E.nF = nF;
    target_faces = std::max<i64>(1, target_faces);

    VecI edges, counts;
    unique_edges(E.F, nV, edges, counts);
    Quadrics qd = build_quadrics(E.P, E.F, edges, counts, o);
    E.Q = std::move(qd.Q);
    E.Qgeo = std::move(qd.Qgeo);
    E.Ageo = std::move(qd.Ageo);
    double t_quadric = now();

    VecI fsrc(E.F.begin(), E.F.end()), fid(3 * nF);
    for (i64 i = 0; i < 3 * nF; i++) fid[i] = i / 3;
    E.L = build_lists(nV, fsrc, fid);

    VecI vpairs;  // (k, 2)
    if (o.virtual_edges) vpairs = find_virtual_edges(E.P, E.F, o.virtual_tau);
    i64 nvp = (i64)vpairs.size() / 2;
    {
        VecI vs, vd;
        for (i64 i = 0; i < nvp; i++) vs.push_back(vpairs[i * 2]);
        for (i64 i = 0; i < nvp; i++) vs.push_back(vpairs[i * 2 + 1]);
        for (i64 i = 0; i < nvp; i++) vd.push_back(vpairs[i * 2 + 1]);
        for (i64 i = 0; i < nvp; i++) vd.push_back(vpairs[i * 2]);
        E.V = build_lists(nV, vs, vd);
    }
    double t_virtual = now();

    // local guarantee data (_lh_data)
    bool lh_on = o.local_tol != 0;
    BVH own_ref;
    if (lh_on) {
        double tol = o.local_tol;
        VecD ptP;
        VecI face;
        for (i64 v = 0; v < nV; v++) {
            if (E.L.ln[v] <= 0) continue;
            for (int k = 0; k < 3; k++) ptP.push_back(P_in[v * 3 + k]);
            face.push_back(E.L.pool[E.L.st[v]]);
        }
        if (nF <= 400000) {
            VecI cf;
            for (i64 f = 0; f < nF; f++)
                if (E.L.ln[F_in[f * 3]] > 0 && E.L.ln[F_in[f * 3 + 1]] > 0 && E.L.ln[F_in[f * 3 + 2]] > 0)
                    cf.push_back(f);
            if (!cf.empty()) {
                VecD c(cf.size() * 3);
                VecI v0(cf.size());
                for (size_t i = 0; i < cf.size(); i++) {
                    i64 f = cf[i];
                    for (int k = 0; k < 3; k++) {
                        double s = 0.0;
                        for (int j = 0; j < 3; j++) s = s + P_in[F_in[f * 3 + j] * 3 + k];
                        c[i * 3 + k] = s / 3;
                    }
                    v0[i] = F_in[f * 3];
                }
                VecI near;
                nearest_incident_face(c, v0, E, near);
                for (size_t i = 0; i < cf.size(); i++) {
                    if (near[i] < 0) continue;
                    for (int k = 0; k < 3; k++) ptP.push_back(c[i * 3 + k]);
                    face.push_back(near[i]);
                }
            }
        }
        i64 nP = (i64)face.size();
        E.K.ptP = std::move(ptP);
        E.K.tol2.assign(nP, tol * tol);
        E.K.fhead.assign(nF, -1);
        E.K.pnext.assign(nP, -1);
        VecI order = np::argsort_stable(face);
        for (i64 i = 0; i + 1 < nP; i++)
            E.K.pnext[order[i]] = face[order[i + 1]] == face[order[i]] ? order[i + 1] : -1;
        for (i64 i = 0; i < nP; i++)
            if (i == 0 || face[order[i]] != face[order[i - 1]]) E.K.fhead[face[order[i]]] = order[i];
        E.K.pbest.assign(nP, 0);
        E.K.vtol2.assign(nV, tol * tol);
        E.K.buf.assign(nP, 0);
        E.K.fan.assign(nF + 16, 0);
        if (!reference) {
            own_ref = BVH(P_in, F_in);
            reference = &own_ref;
        }
        E.K.ref = view(*reference);
    }

    if (lh_on && collapse_parallel() && num_threads() > 1) {
        E.nthreads = num_threads();
        E.scratch.resize(E.nthreads);
    }
    E.prm[0] = o.w_area;
    E.prm[1] = o.flip_threshold;
    E.prm[2] = o.max_move_factor;
    E.prm[3] = np::py_pow(o.min_edge_rel, 2.0);
    E.prm[4] = 1.0 / (qd.mean_edge * qd.mean_edge);
    E.prm[5] = o.preserve_topology ? 1.0 : 0.0;
    E.prm[6] = 1e-3;
    E.prm[7] = o.topology_penalty;
    E.prm[8] = o.max_error ? np::py_pow(o.max_error, 2.0) : INF;
    E.prm[9] = lh_on ? 1.0 : 0.0;
    E.mark.assign(nV, -1);
    E.cnt.assign(nV, 0);
    E.mark2.assign(nV, -1);
    E.rbA.assign(nV + 1, 0);
    E.rbB.assign(nV + 1, 0);
    E.stamp.assign(nV, 0);
    E.cA.assign(nV, 0);
    E.oppc.assign(nV, 0);
    E.H.H.assign(((i64)((double)counts.size() * 1.5) + 64) * 8, 0.0);
    E.H.hs = 0;
    {
        double S[64] = {0}, best[3] = {0};
        for (size_t e = 0; e < counts.size(); e++) E.push_edge(edges[e * 2], edges[e * 2 + 1], S, best);
        for (i64 e = 0; e < nvp; e++) E.push_edge(vpairs[e * 2], vpairs[e * 2 + 1], S, best);
    }
    double t_queue = now();

    E.face_alive.assign(nF, 1);
    E.v_alive.assign(nV, 1);
    E.parent.resize(nV);
    for (i64 i = 0; i < nV; i++) E.parent[i] = i;
    E.C[C_FACES] = nF;
    E.hist.assign(nV * 7, 0.0);
    VecD ls = np::linspace((double)nF, (double)target_faces, 40);
    VecI steps;
    for (double v : ls) steps.push_back((i64)v);
    std::sort(steps.begin(), steps.end());
    steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
    std::reverse(steps.begin(), steps.end());
    i64 rebuilds = o.auto_mode ? 1 : (i64)o.max_rebuilds;
    for (i64 stop : steps) {
        E.run_collapses(target_faces, stop, rebuilds);
        if (E.C[C_DONE] || E.C[C_FACES] <= target_faces) break;
    }
    double t_loop = now();

    // compact output
    SimplifyResult R;
    std::vector<char> used_v(nV, 0);
    for (i64 f = 0; f < nF; f++)
        if (E.face_alive[f])
            for (int k = 0; k < 3; k++) used_v[E.F[f * 3 + k]] = 1;
    VecI new_index(nV, -1);
    i64 nu = 0;
    for (i64 v = 0; v < nV; v++)
        if (used_v[v]) new_index[v] = nu++;
    for (i64 f = 0; f < nF; f++)
        if (E.face_alive[f])
            for (int k = 0; k < 3; k++) R.faces.push_back(new_index[E.F[f * 3 + k]]);
    for (i64 v = 0; v < nV; v++)
        if (used_v[v])
            for (int k = 0; k < 3; k++) R.positions.push_back(E.P[v * 3 + k]);
    VecI root = E.parent;
    while (true) {
        VecI nxt(nV);
        for (i64 v = 0; v < nV; v++) nxt[v] = root[root[v]];
        if (nxt == root) break;
        root = nxt;
    }
    R.vertex_map.resize(nV);
    for (i64 v = 0; v < nV; v++) R.vertex_map[v] = new_index[root[v]];
    R.history.assign(E.hist.begin(), E.hist.begin() + std::min<i64>(E.C[C_COLL], nV) * 7);

    i64 out_faces = (i64)R.faces.size() / 3;
    R.stats = {{"input_faces", (double)nF},
               {"input_vertices", (double)nV},
               {"output_faces", (double)out_faces},
               {"output_vertices", (double)(R.positions.size() / 3)},
               {"collapses", (double)E.C[C_COLL]},
               {"rejected_flip", (double)E.C[C_FLIP]},
               {"rejected_link", (double)E.C[C_LINK]},
               {"rejected_duplicate", (double)E.C[C_DUP]},
               {"rejected_error", (double)E.C[C_ERR]},
               {"stale", (double)E.C[C_STALE]},
               {"rebuilds", (double)E.C[C_REBUILD]},
               {"virtual_edges", (double)nvp},
               {"virtual_collapses", (double)E.C[C_VIRT]},
               {"boundary_vertices", (double)qd.n_boundary_vertices},
               {"reached_target", out_faces <= target_faces ? 1.0 : 0.0},
               {"time_quadrics", t_quadric - t0},
               {"time_virtual", t_virtual - t_quadric},
               {"time_queue", t_queue - t_virtual},
               {"time_collapse", t_loop - t_queue},
               {"time_total", now() - t0}};
    return R;
}

}  // namespace faqem
