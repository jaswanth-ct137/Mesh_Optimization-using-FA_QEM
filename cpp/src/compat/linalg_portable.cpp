// Portable build: the numpy/BLAS operations of accelerate.hpp as fixed-order scalar code, so the
// result depends only on IEEE-754 arithmetic (identical on macOS, Linux and Windows).
// (The Python-parity build uses accelerate.cpp, which replays numpy's Apple Accelerate calls.)
#include <cmath>

#include "accelerate.hpp"

namespace faqem {
namespace accel {

Mat4 Mat4::eye() {
    Mat4 I;
    for (int i = 0; i < 16; i++) I.m[i] = (i % 5 == 0) ? 1.0 : 0.0;
    return I;
}

// sum_k a_k * b_k, left to right
static inline double dot(const double* a, int sa, const double* b, int sb, int n) {
    double s = a[0] * b[0];
    for (int k = 1; k < n; k++) s = s + a[k * sa] * b[k * sb];
    return s;
}

Mat4 dot4(const Mat4& A, const Mat4& B) {
    Mat4 C;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) C(r, c) = dot(&A.m[r * 4], 1, &B.m[c], 4, 4);
    C.forder = false;
    return C;
}

VecD transform_points(const VecD& V, const Mat4& T) {
    size_t n = V.size() / 3;
    VecD out(V.size());
    for (size_t i = 0; i < n; i++)
        for (int k = 0; k < 3; k++)
            out[i * 3 + k] = ((V[i * 3] * T(k, 0) + V[i * 3 + 1] * T(k, 1)) + V[i * 3 + 2] * T(k, 2)) + T(k, 3);
    return out;
}

bool det3_negative(const Mat4& T) {
    double c0 = T(1, 1) * T(2, 2) - T(1, 2) * T(2, 1);
    double c1 = T(1, 0) * T(2, 2) - T(1, 2) * T(2, 0);
    double c2 = T(1, 0) * T(2, 1) - T(1, 1) * T(2, 0);
    double det = (T(0, 0) * c0 - T(0, 1) * c1) + T(0, 2) * c2;
    return det < 0;
}

// closest rotation to a nearly orthonormal R: the orthogonal polar factor (== U V^T of the SVD for
// a non-singular R), by Newton's iteration X <- (X + X^-T) / 2 with a fixed iteration count.
static void polar3(const double R[9], double X[9]) {
    for (int i = 0; i < 9; i++) X[i] = R[i];
    for (int it = 0; it < 12; it++) {
        // inverse transpose: cofactor matrix / det
        double c[9];
        c[0] = X[4] * X[8] - X[5] * X[7];
        c[1] = X[5] * X[6] - X[3] * X[8];
        c[2] = X[3] * X[7] - X[4] * X[6];
        c[3] = X[2] * X[7] - X[1] * X[8];
        c[4] = X[0] * X[8] - X[2] * X[6];
        c[5] = X[1] * X[6] - X[0] * X[7];
        c[6] = X[1] * X[5] - X[2] * X[4];
        c[7] = X[2] * X[3] - X[0] * X[5];
        c[8] = X[0] * X[4] - X[1] * X[3];
        double det = (X[0] * c[0] + X[1] * c[1]) + X[2] * c[2];
        if (det == 0) return;
        for (int i = 0; i < 9; i++) X[i] = 0.5 * (X[i] + c[i] / det);
    }
}

Mat4 fix_rigid(const Mat4& M, double max_deviance) {
    double check = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double s = dot(&M.m[i * 4], 1, &M.m[j * 4], 1, 3);
            check = np::fmax0(check, std::fabs(s - (i == j ? 1.0 : 0.0)));
        }
    if (!(check > 1e-13 && check < max_deviance)) return M;
    double R[9], X[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) R[i * 3 + j] = M(i, j);
    polar3(R, X);
    Mat4 out = Mat4::eye();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) out(i, j) = X[i * 3 + j];
    for (int i = 0; i < 3; i++) out(i, 3) = M(i, 3);
    return out;
}

Mat4 quaternion_matrix(const double qin[4]) {
    double q[4] = {qin[0], qin[1], qin[2], qin[3]};
    double nrm = ((q[0] * q[0] + q[1] * q[1]) + q[2] * q[2]) + q[3] * q[3];
    const double EPS = 2.220446049250313e-16 * 4.0;
    if (nrm < EPS) return Mat4::eye();
    double sc = std::sqrt(2.0 / nrm);
    for (int i = 0; i < 4; i++) q[i] *= sc;
    auto o = [&](int i, int j) { return q[j] * q[i]; };
    Mat4 R;
    for (int i = 0; i < 16; i++) R.m[i] = 0.0;
    R(0, 0) = 1.0 - o(2, 2) - o(3, 3);
    R(0, 1) = o(1, 2) - o(3, 0);
    R(0, 2) = o(1, 3) + o(2, 0);
    R(1, 0) = o(1, 2) + o(3, 0);
    R(1, 1) = 1.0 - o(1, 1) - o(3, 3);
    R(1, 2) = o(2, 3) - o(1, 0);
    R(2, 0) = o(1, 3) - o(2, 0);
    R(2, 1) = o(2, 3) + o(1, 0);
    R(2, 2) = 1.0 - o(1, 1) - o(2, 2);
    R(3, 3) = 1.0;
    R.forder = false;
    return R;
}

VecD rowdot_ones(const VecD& A, int64_t n, int k) {
    VecD y(n, 0.0);
    for (int64_t i = 0; i < n; i++) {
        double s = A[i * k];
        for (int j = 1; j < k; j++) s = s + A[i * k + j];
        y[i] = s;
    }
    return y;
}

VecD unitize_rows(const VecD& v, std::vector<char>* valid) {
    int64_t n = (int64_t)v.size() / 3;
    VecD out(v.size());
    if (valid) valid->assign(n, 0);
    const double TOL_ZERO = 1e-15 * 100;
    for (int64_t i = 0; i < n; i++) {
        double nr = std::sqrt((v[i * 3] * v[i * 3] + v[i * 3 + 1] * v[i * 3 + 1]) + v[i * 3 + 2] * v[i * 3 + 2]);
        bool ok = nr > TOL_ZERO;
        if (ok) nr = 1.0 / nr;
        if (valid) (*valid)[i] = ok;
        for (int k = 0; k < 3; k++) out[i * 3 + k] = v[i * 3 + k] * nr;
    }
    return out;
}

}  // namespace accel
}  // namespace faqem
