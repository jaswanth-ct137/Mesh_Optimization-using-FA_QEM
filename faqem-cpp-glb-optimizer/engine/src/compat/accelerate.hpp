// numpy-on-Accelerate replicas: the BLAS/LAPACK calls numpy 2.2.6 makes on macOS arm64, issued
// with the same arguments so results are bit-identical (used by the trimesh loader emulation).
#pragma once
#include "npcompat.hpp"

namespace faqem {
namespace accel {

// a numpy (4, 4) float64 matrix: values (row-major) + memory layout of the numpy array
struct alignas(16) Mat4 {
    double m[16];
    bool forder = false;  // numpy array is Fortran-contiguous (a transposed view)
    static Mat4 eye();
    double& operator()(int r, int c) { return m[r * 4 + c]; }
    double operator()(int r, int c) const { return m[r * 4 + c]; }
};

// np.dot(A, B) for (4, 4) float64 (Accelerate dgemm; result C-contiguous)
Mat4 dot4(const Mat4& A, const Mat4& B);
// faqem/io.py: V @ T[:3, :3].T + T[:3, 3] for V (n, 3)
VecD transform_points(const VecD& V, const Mat4& T);
// np.linalg.det(T[:3, :3]) < 0
bool det3_negative(const Mat4& T);
// trimesh.transformations.fix_rigid(matrix, max_deviance)
Mat4 fix_rigid(const Mat4& M, double max_deviance = 1e-5);
// np.dot(A, [1.0] * k) for a C-contiguous (n, k) float64 array (cblas_dgemv, as numpy calls it)
VecD rowdot_ones(const VecD& A, int64_t n, int k);
// trimesh.util.unitize for (n, 3) rows: norm = sqrt(np.dot(v * v, [1, 1, 1])); valid if > 1e-13
VecD unitize_rows(const VecD& v, std::vector<char>* valid = nullptr);
// trimesh.transformations.quaternion_matrix(wxyz)
Mat4 quaternion_matrix(const double q[4]);

}  // namespace accel
}  // namespace faqem
