#include "pipeline.hpp"

#include <chrono>
#include <filesystem>

#include "io/export.hpp"

namespace faqem {

const double WIRE_ON_CLAY[4] = {0.04, 0.06, 0.16, 1.0};    // dark navy lines on the grey clay model
const double WIRE_ON_TEXTURE[4] = {0.0, 0.85, 1.0, 1.0};   // cyan lines on textures

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

LoadedMesh::LoadedMesh(const std::string& p) : LoadedMesh(p, Asset()) {}

LoadedMesh::LoadedMesh(const std::string& p, Asset a) : path(p) {
    double t = now();
    asset = a.positions.empty() ? load_asset(p) : std::move(a);
    prep = prepare_mesh(asset.positions, asset.faces);
    stats = mesh_stats(prep.nV(), prep.faces);
    load_time = now() - t;
}

const BVH& LoadedMesh::bvh() {
    if (!bvh_) bvh_.reset(new BVH(prep.positions, prep.faces));
    return *bvh_;
}

static std::string thousands(i64 v) {
    std::string s = std::to_string(v), o;
    for (size_t i = 0; i < s.size(); i++) {
        if (i && (s.size() - i) % 3 == 0) o += ',';
        o += s[i];
    }
    return o;
}

static void mkdirs(const std::string& d) { std::filesystem::create_directories(d); }

static std::string stem(const std::string& path) {
    size_t s = path.find_last_of('/');
    std::string b = s == std::string::npos ? path : path.substr(s + 1);
    size_t d = b.find_last_of('.');
    return (d == std::string::npos || d == 0) ? b : b.substr(0, d);
}

RunOutput run(LoadedMesh& mesh, const std::string& out_dir, const RunOptions& ro, const Progress& progress) {
    mkdirs(out_dir);
    const PreparedMesh& prep = mesh.prep;
    const Asset& asset = mesh.asset;
    auto report = [&](double f, const std::string& msg) {
        if (progress) progress(f, msg);
    };
    RunOutput out;
    bool have_auto_metrics = false;
    if (ro.auto_tolerance) {
        report(0.02, "Automatic: simplifying while keeping detail within the limit");
        SearchResult s = simplify_auto(prep, mesh.bvh(), ro.auto_tolerance, ro.options, 4, [&](int k, i64 n) {
            report(0.05 + 0.1 * k + 0.08, "Automatic pass " + std::to_string(k + 1) + ": " + thousands(n) + " faces");
        });
        out.res = std::move(s.res);
        out.metrics = s.metrics;
        out.passes = s.passes;
        out.auto_mode = true;
        out.auto_within = s.within;
        have_auto_metrics = true;
    } else if (!ro.fast) {
        i64 target = std::min(std::max<i64>(ro.target_faces, 4), prep.nF());
        report(0.02, "Best result for this face count: searching");
        SearchResult s = simplify_to_count(prep, mesh.bvh(), target, ro.options, 6, [&](int k, i64 n) {
            report(0.05 + 0.09 * k + 0.08, "Pass " + std::to_string(k + 1) + ": " + thousands(n) + " faces");
        });
        out.res = std::move(s.res);
        out.metrics = s.metrics;
        out.passes = s.passes;
        out.best_for_count = true;
        out.plain_hausdorff = s.plain_hausdorff;
        have_auto_metrics = true;
    } else {
        i64 target = std::min(std::max<i64>(ro.target_faces, 4), prep.nF());
        report(0.02, "Simplifying (FA-QEM)");
        out.res = simplify(prep.positions, prep.faces, target, ro.options);
    }
    out.time_total = out.res.stats.count("time_total") ? out.res.stats["time_total"] : 0;
    const VecD& S = out.res.positions;
    const VecI& SF = out.res.faces;
    std::string base = stem(mesh.path);
    std::string dir = out_dir + "/";

    VecD Sm = prep.to_model(S);
    out.geometry_obj = export_geometry(dir + base + "_lowpoly.obj", Sm, SF);
    VecD vnorm = trimesh_vertex_normals(Sm, SF);
    out.geometry_glb = export_geometry(dir + base + "_lowpoly_geometry.glb", Sm, SF, &vnorm);

    double t = now();
    if (ro.bake_color || ro.bake_normal) {
        report(0.7, "Baking appearance (successive mapping)");
        BakeResult b = bake(S, SF, prep, asset, out.res.vertex_map, ro.atlas_size, ro.bake_color, ro.bake_normal, true,
                            ro.crease_angle, &mesh.bvh());
        const Material& mat0 = asset.materials[0];
        out.textured_glb = export_textured(dir + base + "_lowpoly_textured.glb", prep.to_model(b.positions), b.normals,
                                           b.uv, b.color.get(), b.normal_map.get(), b.mr.get(), b.base_color,
                                           b.metallic, b.roughness, mat0.double_sided, mat0.alpha_mode);
        out.atlas_size = b.atlas_size;
        out.baked_color = (bool)b.color;
        out.baked_normal = (bool)b.normal_map;
    }
    out.time_bake = now() - t;

    if (have_auto_metrics) {
        out.has_metrics = true;
    } else if (ro.compute_metrics) {
        report(0.9, "Measuring Hausdorff / Chamfer error");
        out.metrics = compare_meshes(prep.positions, prep.faces, S, SF, 50000, &mesh.bvh());
        out.has_metrics = true;
    }
    out.clay_wire_glb = add_wireframe(out.geometry_glb, Sm, SF, dir + base + "_lowpoly_wireframe.glb", WIRE_ON_CLAY);
    out.lines_glb = add_wireframe("", Sm, SF, dir + base + "_lowpoly_lines.glb", WIRE_ON_CLAY);
    if (!out.textured_glb.empty())
        out.textured_wire_glb = add_wireframe(out.textured_glb, Sm, SF, dir + base + "_lowpoly_textured_wireframe.glb",
                                              out.baked_color ? WIRE_ON_TEXTURE : WIRE_ON_CLAY);
    report(1.0, "Done");
    return out;
}

}  // namespace faqem
