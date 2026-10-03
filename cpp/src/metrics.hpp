// Hausdorff / Chamfer metrics (port of faqem/metrics.py).
#pragma once
#include <functional>

#include "bvh.hpp"
#include "parallel.hpp"

namespace faqem {

struct Metrics {
    double hausdorff = 0, chamfer = 0, mean_distance = 0;
};

VecD sample_surface(const VecD& P, const VecI& F, i64 n, uint64_t seed, bool include_vertices = true);
VecD point_to_mesh_d2(const VecD& pts, const BVH& bvh);
Metrics compare_meshes(const VecD& PA, const VecI& FA, const VecD& PB, const VecI& FB, i64 samples = 50000,
                       const BVH* bvh_a = nullptr);


}  // namespace faqem
