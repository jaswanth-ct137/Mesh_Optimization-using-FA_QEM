// numpy-compatible numerics: every helper reproduces the exact floating-point evaluation order
// of the numpy 2.2.6 routine it replaces (macOS arm64 build), so results are bit-identical.
// The whole project is compiled with -ffp-contract=off; FMA appears only where numpy itself uses it.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace faqem {

using i64 = int64_t;
using VecD = std::vector<double>;
using VecI = std::vector<int64_t>;

namespace np {

// ---- scalar semantics -----------------------------------------------------------------------
// Python float ** and numpy array ** (exponent not in {-1, 0, 0.5, 1, 2}) call pow; this goes
// through fm::pow (libm in the Python-parity build, correctly rounded CORE-MATH otherwise) and is
// never folded into x * x.
double py_pow(double x, double y);

// numpy elementwise minimum/maximum and min()/max() reductions: -0.0 orders below +0.0
inline double fmin0(double a, double b) {
    if (a < b) return a;
    if (b < a) return b;
    if (a == b) return std::signbit(a) ? a : b;
    return std::isnan(a) ? a : b;  // NaN propagates
}
inline double fmax0(double a, double b) {
    if (a > b) return a;
    if (b > a) return b;
    if (a == b) return std::signbit(a) ? b : a;
    return std::isnan(a) ? a : b;
}
// np.clip(x, lo, hi) on float64
inline double clip(double x, double lo, double hi) {
    double t = (x < lo) ? lo : x;
    return (t > hi) ? hi : t;
}
// np.round / np.rint: round half to even
inline double rint(double x) { return std::nearbyint(x); }

// Python / Numba integer floor modulo and floor division
inline i64 floor_mod(i64 a, i64 b) {
    i64 r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}
inline i64 floor_div(i64 a, i64 b) {
    i64 q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

// ---- reductions -------------------------------------------------------------------------------
// numpy pairwise summation (loops_utils.h.src) of n doubles with a stride
double pairwise_sum(const double* a, i64 n, i64 stride = 1);
// np.add.reduce over a 1-D float64 array: identity 0, buffered in chunks of 8192 elements
double sum(const double* a, i64 n, i64 stride = 1);
inline double sum(const VecD& a) { return sum(a.data(), (i64)a.size()); }
inline double mean(const VecD& a) { return sum(a) / (double)a.size(); }
// x.max() / x.min() reductions (NaN-free inputs)
double vmax(const double* a, i64 n);
double vmin(const double* a, i64 n);

// np.linalg.norm(x, axis=1) for an (n, 3) row: sqrt((x0^2 + x1^2) + x2^2)
inline double norm3(double x, double y, double z) { return std::sqrt((x * x + y * y) + z * z); }
// np.einsum('ij,ij->i') on C-contiguous (n, 3) float64 (NEON path): fma(a2, b2, a0*b0) + a1*b1
inline double einsum3(double a0, double a1, double a2, double b0, double b1, double b2) {
    return std::fma(a2, b2, a0 * b0) + a1 * b1;
}
// np.cross component order
inline void cross(double ax, double ay, double az, double bx, double by, double bz,
                  double& cx, double& cy, double& cz) {
    cx = ay * bz - az * by;
    cy = az * bx - ax * bz;
    cz = ax * by - ay * bx;
}

// np.linspace(start, stop, num) (endpoint=True)
VecD linspace(double start, double stop, i64 num);

// ---- sorting ----------------------------------------------------------------------------------
// np.argsort default (introsort, numpy quicksort.cpp, unstable) for int64 / float64 keys
VecI argsort_quick(const VecI& v);
VecI argsort_quick(const VecD& v);
// np.argsort(kind="stable")
template <typename T>
VecI argsort_stable(const std::vector<T>& v) {
    VecI idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](i64 a, i64 b) { return v[a] < v[b]; });
    return idx;
}

// np.unique(rows, axis=0, return_index=True, return_inverse=True, return_counts=True) on an
// (n, k) int64 array: rows in signed lexicographic order; index = first occurrence.
struct UniqueRows {
    VecI rows;     // (u, k) unique rows
    VecI index;    // (u,) first occurrence
    VecI inverse;  // (n,)
    VecI counts;   // (u,)
};
UniqueRows unique_rows(const i64* a, i64 n, int k);

// np.searchsorted(a, v, side) on a sorted array
inline i64 searchsorted_left(const VecI& a, i64 v) {
    return (i64)(std::lower_bound(a.begin(), a.end(), v) - a.begin());
}

}  // namespace np
}  // namespace faqem
