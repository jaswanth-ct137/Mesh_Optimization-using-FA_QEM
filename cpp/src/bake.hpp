// Stage 2 - appearance transfer via successive mapping (port of faqem/bake.py).
#pragma once
#include "bvh.hpp"
#include "io/asset.hpp"
#include "prep.hpp"

namespace faqem {

struct BakeResult {
    VecD positions;  // (3m, 3) per-corner, normalised space
    VecD normals;    // (3m, 3)
    VecD uv;         // (3m, 2)
    ImagePtr color, normal_map, mr;  // null when not baked
    int atlas_size = 0;
    i64 pair_charts = 0;
    double texels_per_unit = 0;
    double base_color[4] = {0.8, 0.8, 0.8, 1.0};
    double metallic = 0.0, roughness = 1.0;
};

// crease-aware corner normals (out (m, 3, 3)) and unit face normals (fu (m, 3))
void corner_normals(const VecD& P, const VecI& F, double crease_cos, VecD& out, VecD& fu);

BakeResult bake(const VecD& S, const VecI& SF, const PreparedMesh& prep, const Asset& asset, const VecI& vertex_map,
                int atlas_size = 0, bool bake_color = true, bool bake_normal = true, bool bake_mr = true,
                double crease_angle = 60.0, const BVH* orig_bvh = nullptr);

}  // namespace faqem
