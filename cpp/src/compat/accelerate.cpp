#define ACCELERATE_NEW_LAPACK
#define ACCELERATE_LAPACK_ILP64
#include "accelerate.hpp"

#include <Accelerate/Accelerate.h>

#include <cmath>

namespace faqem {
namespace accel {

Mat4 Mat4::eye() {
    Mat4 I;
    for (int i = 0; i < 16; i++) I.m[i] = (i % 5 == 0) ? 1.0 : 0.0;
    return I;
}

// numpy dot of two C-contiguous / F-contiguous (4, 4) matrices -> cblas_dgemm RowMajor
Mat4 dot4(const Mat4& A, const Mat4& B) {
    // memory buffers in the numpy layout
    alignas(16) double a[16];
    alignas(16) double b[16];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            if (A.forder) a[c * 4 + r] = A(r, c);
            else a[r * 4 + c] = A(r, c);
            if (B.forder) b[c * 4 + r] = B(r, c);
            else b[r * 4 + c] = B(r, c);
        }
    Mat4 C;
    C.forder = false;
    cblas_dgemm(CblasRowMajor, A.forder ? CblasTrans : CblasNoTrans, B.forder ? CblasTrans : CblasNoTrans, 4, 4, 4,
                1.0, a, 4, b, 4, 0.0, C.m, 4);
    return C;
}

VecD transform_points(const VecD& V, const Mat4& T) {
    __LAPACK_int n = (__LAPACK_int)(V.size() / 3);
    VecD out(V.size());
    if (n == 0) return out;
    // the numpy buffer of T (row-major 4x4 if C order, column-major if F order)
    alignas(16) double t[16];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            if (T.forder) t[c * 4 + r] = T(r, c);
            else t[r * 4 + c] = T(r, c);
        }
    // B = T[:3, :3].T: C-order T -> strides (8, 32) -> Trans, ldb 4; F-order -> NoTrans, ldb 4
    if (n >= 2) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, T.forder ? CblasNoTrans : CblasTrans, n, 3, 3, 1.0, V.data(), 3, t,
                    4, 0.0, out.data(), 3);
    } else {
        // vector @ matrix -> gemv (numpy matmul special case)
        cblas_dgemv(T.forder ? CblasRowMajor : CblasColMajor, CblasTrans, 3, 3, 1.0, t, 4, V.data(), 1, 0.0,
                    out.data(), 1);
    }
    for (__LAPACK_int i = 0; i < n; i++)
        for (int k = 0; k < 3; k++) out[i * 3 + k] = out[i * 3 + k] + T(k, 3);
    return out;
}

bool det3_negative(const Mat4& T) {
    // numpy det: LAPACK getrf on the Fortran-ordered copy, product of the diagonal x permutation sign
    alignas(16) double a[9];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) a[c * 3 + r] = T(r, c);
    __LAPACK_int n = 3, lda = 3, info = 0;
    __LAPACK_int ipiv[3];
    dgetrf_(&n, &n, a, &lda, ipiv, &info);
    double sign = 1.0, acc = 1.0;
    for (int i = 0; i < 3; i++) {
        if (ipiv[i] != i + 1) sign = -sign;
        double d = a[i * 3 + i];
        if (d < 0) {
            sign = -sign;
            d = -d;
        }
        acc *= d;
    }
    double det = info > 0 ? 0.0 : sign * acc;
    return det < 0;
}

Mat4 fix_rigid(const Mat4& M, double max_deviance) {
    // check = abs(dot(R, R.T) - I).max(): same buffer with its transpose -> numpy uses syrk
    alignas(16) double r[16];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            if (M.forder) r[j * 4 + i] = M(i, j);
            else r[i * 4 + j] = M(i, j);
        }
    alignas(16) double c[9] = {0};
    // R = matrix[:3, :3] view: C order -> lda 4 NoTrans; F order -> the view is (Trans of) lda 4
    cblas_dsyrk(CblasRowMajor, CblasUpper, M.forder ? CblasTrans : CblasNoTrans, 3, 3, 1.0, r, 4, 0.0, c, 3);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < i; j++) c[i * 3 + j] = c[j * 3 + i];
    double check = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) check = np::fmax0(check, std::fabs(c[i * 3 + j] - (i == j ? 1.0 : 0.0)));
    if (!(check > 1e-13 && check < max_deviance)) return M;
    // U, _, V = np.linalg.svd(R) (gesdd, jobz 'A', Fortran-ordered copy). The buffers are laid out
    // exactly like numpy's init_gesdd (one malloc: a | s | u | vt | iwork, then a separate work
    // block): Accelerate's kernels take alignment-dependent paths, so the layout changes the bits.
    char jobz = 'A';
    __LAPACK_int m = 3, n = 3, lda = 3, ldu = 3, ldvt = 3, info = 0, lwork = -1;
    const size_t a_size = 9 * sizeof(double), s_size = 3 * sizeof(double), u_size = 9 * sizeof(double),
                 vt_size = 9 * sizeof(double), iwork_size = 8 * 3 * sizeof(__LAPACK_int);
    uint8_t* mem = (uint8_t*)std::malloc(a_size + s_size + u_size + vt_size + iwork_size);
    double* a = (double*)mem;
    double* s = (double*)(mem + a_size);
    double* u = (double*)(mem + a_size + s_size);
    double* vt = (double*)(mem + a_size + s_size + u_size);
    __LAPACK_int* iwork = (__LAPACK_int*)(mem + a_size + s_size + u_size + vt_size);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) a[j * 3 + i] = M(i, j);
    double wq = 0;
    dgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wq, &lwork, iwork, &info);
    lwork = (__LAPACK_int)wq;
    if (lwork == 0) lwork = 1;
    double* work = (double*)std::malloc((size_t)lwork * sizeof(double));
    dgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info);
    std::free(work);
    // delinearize to C-order U, V
    alignas(16) double U[9];
    alignas(16) double Vv[9];
    alignas(16) double UV[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            U[i * 3 + j] = u[j * 3 + i];
            Vv[i * 3 + j] = vt[j * 3 + i];
        }
    std::free(mem);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, U, 3, Vv, 3, 0.0, UV, 3);
    Mat4 out = Mat4::eye();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) out(i, j) = UV[i * 3 + j];
    for (int i = 0; i < 3; i++) out(i, 3) = M(i, 3);
    return out;
}

VecD rowdot_ones(const VecD& A, int64_t n, int k) {
    VecD y(n, 0.0);
    if (n == 0) return y;
    std::vector<double> ones(k, 1.0);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, (__LAPACK_int)n, (__LAPACK_int)k, 1.0, A.data(), (__LAPACK_int)k,
                ones.data(), 1, 0.0, y.data(), 1);
    return y;
}

VecD unitize_rows(const VecD& v, std::vector<char>* valid) {
    int64_t n = (int64_t)v.size() / 3;
    VecD sq(v.size());
    for (size_t i = 0; i < v.size(); i++) sq[i] = v[i] * v[i];
    VecD norm = rowdot_ones(sq, n, 3);
    VecD out(v.size());
    if (valid) valid->assign(n, 0);
    const double TOL_ZERO = 1e-15 * 100;  // trimesh TOL_ZERO = np.finfo(float64).resolution * 100
    for (int64_t i = 0; i < n; i++) {
        double nr = std::sqrt(norm[i]);
        bool ok = nr > TOL_ZERO;
        if (ok) nr = 1.0 / nr;
        if (valid) (*valid)[i] = ok;
        for (int k = 0; k < 3; k++) out[i * 3 + k] = v[i * 3 + k] * nr;
    }
    return out;
}

Mat4 quaternion_matrix(const double qin[4]) {
    double q[4] = {qin[0], qin[1], qin[2], qin[3]};
    // n = np.dot(q * q, [1.0] * 4)  (gemv)
    alignas(16) double qq[4];
    alignas(16) double ones[4] = {1.0, 1.0, 1.0, 1.0};
    alignas(16) double nrm = 0.0;
    for (int i = 0; i < 4; i++) qq[i] = q[i] * q[i];
    cblas_dgemv(CblasRowMajor, CblasNoTrans, 1, 4, 1.0, qq, 4, ones, 1, 0.0, &nrm, 1);
    const double EPS = 2.220446049250313e-16 * 4.0;
    Mat4 R;
    if (nrm < EPS) return Mat4::eye();
    double sc = std::sqrt(2.0 / nrm);
    for (int i = 0; i < 4; i++) q[i] *= sc;
    auto o = [&](int i, int j) { return q[j] * q[i]; };  // q[:, None, :] * q[:, :, None] -> [i][j] = q[j]*q[i]
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

}  // namespace accel
}  // namespace faqem
