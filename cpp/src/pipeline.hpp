// End-to-end pipeline: load -> prepare -> simplify -> bake -> export (port of faqem/pipeline.py).
#pragma once
#include <functional>
#include <memory>
#include <string>

#include "bake.hpp"
#include "core.hpp"
#include "io/asset.hpp"
#include "metrics.hpp"
#include "prep.hpp"
#include "search.hpp"

namespace faqem {

struct LoadedMesh {
    std::string path;
    Asset asset;
    PreparedMesh prep;
    MeshStats stats;
    double load_time = 0;
    explicit LoadedMesh(const std::string& path);
    LoadedMesh(const std::string& path, Asset asset);
    const BVH& bvh();

private:
    std::unique_ptr<BVH> bvh_;
};

struct RunOptions {
    Options options;
    bool bake_color = true, bake_normal = true;
    int atlas_size = 0;
    double crease_angle = 60.0;
    bool compute_metrics = true;
    double auto_tolerance = 0.0;  // > 0 selects Automatic mode
    bool fast = false;
    i64 target_faces = 0;
};

struct RunOutput {
    SimplifyResult res;
    std::vector<SearchPass> passes;
    bool best_for_count = false, auto_mode = false, auto_within = false;
    double plain_hausdorff = 0;
    bool has_metrics = false;
    Metrics metrics;
    double time_total = 0, time_bake = 0;
    int atlas_size = 0;
    bool baked_color = false, baked_normal = false;
    std::string geometry_obj, geometry_glb, textured_glb, clay_wire_glb, lines_glb, textured_wire_glb;
};

using Progress = std::function<void(double, const std::string&)>;

RunOutput run(LoadedMesh& mesh, const std::string& out_dir, const RunOptions& ro, const Progress& progress = nullptr);

extern const double WIRE_ON_CLAY[4];
extern const double WIRE_ON_TEXTURE[4];

}  // namespace faqem
