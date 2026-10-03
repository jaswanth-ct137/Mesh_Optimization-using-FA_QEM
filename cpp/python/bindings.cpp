// pybind11 module faqem_cpp: the C++ FA-QEM engine, with the same entry points as the Python package.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <set>

#include "compat/accelerate.hpp"
#include "compat/csgraph.hpp"
#include "compat/fmath.hpp"
#include "compat/nprandom.hpp"
#include "core.hpp"
#include "io/asset.hpp"
#include "io/export.hpp"
#include "bake.hpp"
#include "parallel.hpp"
#include "pipeline.hpp"
#include "metrics.hpp"
#include "prep.hpp"
#include "search.hpp"
#include "virtual_edges.hpp"

namespace py = pybind11;
using namespace faqem;

using ArrD = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ArrI = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

static VecD vecd(const ArrD& a) { return VecD(a.data(), a.data() + a.size()); }
static VecI veci(const ArrI& a) { return VecI(a.data(), a.data() + a.size()); }

template <typename T>
static py::array_t<T> arr(const std::vector<T>& v, std::vector<py::ssize_t> shape) {
    py::array_t<T> out(shape);
    if (!v.empty()) std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(T));
    return out;
}
static py::array_t<double> arr3(const VecD& v) { return arr(v, {(py::ssize_t)(v.size() / 3), 3}); }
static py::array_t<int64_t> arr3(const VecI& v) { return arr(v, {(py::ssize_t)(v.size() / 3), 3}); }

static Options options_from(const py::object& obj) {
    Options o;
    if (obj.is_none()) return o;
    py::dict d = obj.cast<py::dict>();
    for (auto item : d) {
        if (item.second.is_none()) continue;
        std::string k = item.first.cast<std::string>();
        double v = item.second.cast<double>();
        if (!o.set(k, v)) throw std::invalid_argument("unknown FA-QEM option: " + k);
    }
    return o;
}

static py::dict stats_dict(const std::map<std::string, double>& s) {
    py::dict d;
    static const std::set<std::string> ints = {"input_faces", "input_vertices", "output_faces", "output_vertices",
                                               "collapses", "rejected_flip", "rejected_link", "rejected_duplicate",
                                               "rejected_error", "stale", "rebuilds", "virtual_edges",
                                               "virtual_collapses", "boundary_vertices"};
    static const std::set<std::string> bools = {"reached_target", "auto", "auto_within", "best_for_count"};
    for (auto& kv : s) {
        if (ints.count(kv.first)) d[py::str(kv.first)] = (int64_t)kv.second;
        else if (bools.count(kv.first)) d[py::str(kv.first)] = kv.second != 0;
        else d[py::str(kv.first)] = kv.second;
    }
    return d;
}

static py::dict result_dict(const SimplifyResult& r) {
    py::dict d;
    d["positions"] = arr3(r.positions);
    d["faces"] = arr3(r.faces);
    d["vertex_map"] = arr(r.vertex_map, {(py::ssize_t)r.vertex_map.size()});
    d["history"] = arr(r.history, {(py::ssize_t)(r.history.size() / 7), 7});
    d["stats"] = stats_dict(r.stats);
    return d;
}

static py::dict metrics_dict(const Metrics& m) {
    py::dict d;
    d["hausdorff"] = m.hausdorff;
    d["chamfer"] = m.chamfer;
    d["mean_distance"] = m.mean_distance;
    return d;
}


static py::object image_array(const ImagePtr& im) {
    if (!im) return py::none();
    py::array_t<uint8_t> a({(py::ssize_t)im->h, (py::ssize_t)im->w, (py::ssize_t)im->channels});
    std::memcpy(a.mutable_data(), im->px.data(), im->px.size());
    return a;
}

static py::dict asset_dict(const Asset& A) {
    py::dict d;
    d["positions"] = arr3(A.positions);
    d["faces"] = arr3(A.faces);
    d["face_material"] = arr(A.face_material, {(py::ssize_t)A.face_material.size()});
    int64_t m = (int64_t)A.faces.size() / 3;
    d["corner_uv"] = A.has_uv ? py::object(arr(A.corner_uv, {m, 3, 2})) : py::object(py::none());
    d["corner_color"] = A.has_col ? py::object(arr(A.corner_color, {m, 3, 4})) : py::object(py::none());
    py::list mats;
    for (auto& mt : A.materials) {
        py::dict md;
        md["color"] = arr(VecD(mt.color, mt.color + 4), {4});
        md["texture"] = image_array(mt.texture);
        md["metallic"] = mt.metallic;
        md["roughness"] = mt.roughness;
        md["mr_texture"] = image_array(mt.mr_texture);
        md["alpha_mode"] = mt.alpha_mode;
        md["double_sided"] = mt.double_sided;
        mats.append(md);
    }
    d["materials"] = mats;
    d["has_texture"] = A.has_texture();
    d["has_appearance"] = A.has_appearance();
    d["has_mr_variation"] = A.has_mr_variation();
    py::dict info;
    info["parts"] = (int64_t)A.info.at("parts");
    info["materials"] = (int64_t)A.info.at("materials");
    info["textured"] = A.info.at("textured") != 0;
    info["vertex_colors"] = A.info.at("vertex_colors") != 0;
    d["info"] = info;
    return d;
}

struct PyPrepared {
    PreparedMesh prep;
    std::unique_ptr<BVH> bvh;
    const BVH& get_bvh() {
        if (!bvh) bvh.reset(new BVH(prep.positions, prep.faces));
        return *bvh;
    }
};


struct PyLoaded {
    std::unique_ptr<LoadedMesh> m;
};

static py::object image_or_none(const ImagePtr& im) { return image_array(im); }

#ifndef FAQEM_MODULE_NAME
#define FAQEM_MODULE_NAME faqem_cpp
#endif
PYBIND11_MODULE(FAQEM_MODULE_NAME, m) {
    m.doc() = "FA-QEM mesh simplification (C++ engine, bit-identical to the Python reference)";

    py::class_<PyPrepared>(m, "PreparedMesh", py::module_local())
        .def_property_readonly("positions", [](PyPrepared& p) { return arr3(p.prep.positions); })
        .def_property_readonly("faces", [](PyPrepared& p) { return arr3(p.prep.faces); })
        .def_property_readonly("face_orig",
                               [](PyPrepared& p) { return arr(p.prep.face_orig, {(py::ssize_t)p.prep.face_orig.size()}); })
        .def_property_readonly("input_to_vertex",
                               [](PyPrepared& p) {
                                   return arr(p.prep.input_to_vertex, {(py::ssize_t)p.prep.input_to_vertex.size()});
                               })
        .def_property_readonly("center",
                               [](PyPrepared& p) { return arr(VecD(p.prep.center, p.prep.center + 3), {3}); })
        .def_property_readonly("diag", [](PyPrepared& p) { return p.prep.diag; })
        .def_property_readonly("removed", [](PyPrepared& p) { return p.prep.removed; })
        .def("to_model", [](PyPrepared& p, ArrD x) { return arr3(p.prep.to_model(vecd(x))); });

    m.def(
        "prepare_mesh",
        [](ArrD P, ArrI F, double weld_tolerance, bool make_manifold) {
            auto* p = new PyPrepared();
            p->prep = prepare_mesh(vecd(P), veci(F), weld_tolerance, make_manifold);
            return p;
        },
        py::arg("positions"), py::arg("faces"), py::arg("weld_tolerance") = 1e-6, py::arg("make_manifold") = true);

    m.def(
        "simplify",
        [](ArrD P, ArrI F, int64_t target, py::object options, PyPrepared* reference) {
            Options o = options_from(options);
            const BVH* ref = reference ? &reference->get_bvh() : nullptr;
            SimplifyResult r;
            {
                py::gil_scoped_release nogil;
                r = simplify(vecd(P), veci(F), target, o, ref);
            }
            return result_dict(r);
        },
        py::arg("P"), py::arg("F"), py::arg("target_faces"), py::arg("options") = py::none(),
        py::arg("reference") = nullptr);

    m.def(
        "simplify_auto",
        [](PyPrepared& mesh, double tolerance, py::object options, int max_passes) {
            Options o = options_from(options);
            const BVH& bvh = mesh.get_bvh();
            SearchResult s;
            {
                py::gil_scoped_release nogil;
                s = simplify_auto(mesh.prep, bvh, tolerance, o, max_passes);
            }
            py::dict r = result_dict(s.res);
            py::list passes;
            for (auto& p : s.passes)
                passes.append(py::dict(py::arg("factor") = p.factor_or_tau, py::arg("faces") = p.faces,
                                       py::arg("hausdorff") = p.hausdorff, py::arg("ok") = p.ok));
            py::dict st = r["stats"];
            st["auto_passes"] = passes;
            return py::make_tuple(r, metrics_dict(s.metrics));
        },
        py::arg("mesh"), py::arg("tolerance"), py::arg("options") = py::none(), py::arg("max_passes") = 4);

    m.def(
        "simplify_to_count",
        [](PyPrepared& mesh, int64_t target, py::object options, int max_passes) {
            Options o = options_from(options);
            const BVH& bvh = mesh.get_bvh();
            SearchResult s;
            {
                py::gil_scoped_release nogil;
                s = simplify_to_count(mesh.prep, bvh, target, o, max_passes);
            }
            py::dict r = result_dict(s.res);
            py::list passes;
            for (auto& p : s.passes) {
                py::object h = std::isnan(p.hausdorff) ? py::object(py::none()) : py::object(py::float_(p.hausdorff));
                passes.append(py::dict(py::arg("tau") = p.factor_or_tau, py::arg("faces") = p.faces,
                                       py::arg("hausdorff") = h, py::arg("reached") = p.ok));
            }
            py::dict st = r["stats"];
            st["count_passes"] = passes;
            return py::make_tuple(r, metrics_dict(s.metrics));
        },
        py::arg("mesh"), py::arg("target_faces"), py::arg("options") = py::none(), py::arg("max_passes") = 6);

    m.def(
        "compare_meshes",
        [](ArrD PA, ArrI FA, ArrD PB, ArrI FB, int64_t samples) {
            Metrics mm;
            VecD pa = vecd(PA), pb = vecd(PB);
            VecI fa = veci(FA), fb = veci(FB);
            {
                py::gil_scoped_release nogil;
                mm = compare_meshes(pa, fa, pb, fb, samples);
            }
            return metrics_dict(mm);
        },
        py::arg("PA"), py::arg("FA"), py::arg("PB"), py::arg("FB"), py::arg("samples") = 50000);

    m.def("sample_surface", [](ArrD P, ArrI F, int64_t n, uint64_t seed, bool include_vertices) {
        return arr3(sample_surface(vecd(P), veci(F), n, seed, include_vertices));
    }, py::arg("P"), py::arg("F"), py::arg("n"), py::arg("seed") = 0, py::arg("include_vertices") = true);

    m.def("build_quadrics", [](ArrD P, ArrI F, py::object options) {
        VecD p = vecd(P);
        VecI f = veci(F);
        VecI edges, counts;
        unique_edges(f, (i64)p.size() / 3, edges, counts);
        Quadrics q = build_quadrics(p, f, edges, counts, options_from(options));
        i64 nV = (i64)p.size() / 3;
        return py::make_tuple(arr(q.Q, {nV, 10}), q.mean_area, q.mean_edge, q.n_boundary_vertices,
                              arr(q.Qgeo, {nV, 10}), arr(q.Ageo, {nV}));
    }, py::arg("P"), py::arg("F"), py::arg("options") = py::none());

    m.def("mesh_stats", [](ArrD P, ArrI F) {
        MeshStats s = mesh_stats((i64)(P.size() / 3), veci(F));
        py::dict d;
        d["vertices"] = s.vertices;
        d["faces"] = s.faces;
        d["edges"] = s.edges;
        d["components"] = s.components;
        d["boundary_edges"] = s.boundary_edges;
        d["non_manifold_edges"] = s.non_manifold_edges;
        return d;
    });


    py::class_<PyLoaded>(m, "LoadedMesh", py::module_local())
        .def(py::init([](const std::string& path) {
                 auto* p = new PyLoaded();
                 py::gil_scoped_release nogil;
                 p->m.reset(new LoadedMesh(path));
                 return p;
             }),
             py::arg("path"))
        .def_property_readonly("path", [](PyLoaded& p) { return p.m->path; })
        .def_property_readonly("load_time", [](PyLoaded& p) { return p.m->load_time; })
        .def_property_readonly("stats", [](PyLoaded& p) {
            const MeshStats& s = p.m->stats;
            py::dict d;
            d["vertices"] = s.vertices;
            d["faces"] = s.faces;
            d["edges"] = s.edges;
            d["components"] = s.components;
            d["boundary_edges"] = s.boundary_edges;
            d["non_manifold_edges"] = s.non_manifold_edges;
            return d;
        })
        .def_property_readonly("asset", [](PyLoaded& p) { return asset_dict(p.m->asset); })
        .def_property_readonly("removed", [](PyLoaded& p) { return p.m->prep.removed; })
        .def_property_readonly("asset_info", [](PyLoaded& p) {
            const Asset& A = p.m->asset;
            py::dict info;
            info["parts"] = (int64_t)A.info.at("parts");
            info["materials"] = (int64_t)A.info.at("materials");
            info["textured"] = A.info.at("textured") != 0;
            info["vertex_colors"] = A.info.at("vertex_colors") != 0;
            return info;
        })
        .def_property_readonly("has_appearance", [](PyLoaded& p) { return p.m->asset.has_appearance(); })
        .def("prepared", [](PyLoaded& p) {
            auto* q = new PyPrepared();
            q->prep = p.m->prep;
            return q;
        })
        .def("bake", [](PyLoaded& p, ArrD S, ArrI SF, ArrI vertex_map, int atlas_size, bool bake_color, bool bake_normal,
                        double crease_angle) {
            BakeResult b;
            VecD s = vecd(S);
            VecI sf = veci(SF), vm = veci(vertex_map);
            {
                py::gil_scoped_release nogil;
                const BVH& bvh = p.m->bvh();
                b = bake(s, sf, p.m->prep, p.m->asset, vm, atlas_size, bake_color, bake_normal, true, crease_angle, &bvh);
            }
            py::dict d;
            d["positions"] = arr3(b.positions);
            d["normals"] = arr3(b.normals);
            d["uv"] = arr(b.uv, {(py::ssize_t)(b.uv.size() / 2), 2});
            d["color"] = image_or_none(b.color);
            d["normal_map"] = image_or_none(b.normal_map);
            d["mr"] = image_or_none(b.mr);
            d["atlas_size"] = b.atlas_size;
            d["pair_charts"] = b.pair_charts;
            d["texels_per_unit"] = b.texels_per_unit;
            d["base_color"] = py::make_tuple(b.base_color[0], b.base_color[1], b.base_color[2], b.base_color[3]);
            d["metallic"] = b.metallic;
            d["roughness"] = b.roughness;
            return d;
        }, py::arg("S"), py::arg("SF"), py::arg("vertex_map"), py::arg("atlas_size") = 0, py::arg("bake_color") = true,
           py::arg("bake_normal") = true, py::arg("crease_angle") = 60.0)
        .def("run", [](PyLoaded& p, py::object target_faces, const std::string& out_dir, py::object options,
                       bool bake_color, bool bake_normal, py::object atlas_size, double crease_angle,
                       bool compute_metrics, py::object auto_tolerance, bool fast, py::object progress) {
            RunOptions ro;
            ro.options = options_from(options);
            ro.target_faces = target_faces.is_none() ? 0 : target_faces.cast<int64_t>();
            ro.bake_color = bake_color;
            ro.bake_normal = bake_normal;
            ro.atlas_size = atlas_size.is_none() ? 0 : atlas_size.cast<int>();
            ro.crease_angle = crease_angle;
            ro.compute_metrics = compute_metrics;
            ro.auto_tolerance = auto_tolerance.is_none() ? 0.0 : auto_tolerance.cast<double>();
            ro.fast = fast;
            Progress cb = nullptr;
            if (!progress.is_none())
                cb = [progress](double f, const std::string& msg) {
                    py::gil_scoped_acquire g;
                    progress(f, msg);
                };
            RunOutput r;
            {
                py::gil_scoped_release nogil;
                r = run(*p.m, out_dir, ro, cb);
            }
            py::dict st = stats_dict(r.res.stats);
            py::list passes;
            for (auto& ps : r.passes) {
                py::object h = std::isnan(ps.hausdorff) ? py::object(py::none()) : py::object(py::float_(ps.hausdorff));
                if (r.auto_mode)
                    passes.append(py::dict(py::arg("factor") = ps.factor_or_tau, py::arg("faces") = ps.faces,
                                           py::arg("hausdorff") = h, py::arg("ok") = ps.ok));
                else
                    passes.append(py::dict(py::arg("tau") = ps.factor_or_tau, py::arg("faces") = ps.faces,
                                           py::arg("hausdorff") = h, py::arg("reached") = ps.ok));
            }
            if (r.auto_mode) {
                st["auto_passes"] = passes;
                st["auto_within"] = r.auto_within;
            }
            if (r.best_for_count) {
                st["count_passes"] = passes;
                st["plain_hausdorff"] = r.plain_hausdorff;
            }
            py::dict o;
            o["stats"] = st;
            {
                const MeshStats& s = p.m->stats;
                py::dict d;
                d["vertices"] = s.vertices;
                d["faces"] = s.faces;
                d["edges"] = s.edges;
                d["components"] = s.components;
                d["boundary_edges"] = s.boundary_edges;
                d["non_manifold_edges"] = s.non_manifold_edges;
                o["input"] = d;
            }
            o["positions"] = arr3(r.res.positions);
            o["faces"] = arr3(r.res.faces);
            o["vertex_map"] = arr(r.res.vertex_map, {(py::ssize_t)r.res.vertex_map.size()});
            o["geometry_obj"] = r.geometry_obj;
            o["geometry_glb"] = r.geometry_glb;
            o["textured_glb"] = r.textured_glb.empty() ? py::object(py::none()) : py::object(py::str(r.textured_glb));
            o["clay_wire_glb"] = r.clay_wire_glb;
            o["lines_glb"] = r.lines_glb;
            if (!r.textured_wire_glb.empty()) o["textured_wire_glb"] = r.textured_wire_glb;
            o["time_bake"] = r.time_bake;
            if (!r.textured_glb.empty()) {
                o["atlas_size"] = r.atlas_size;
                o["baked_color"] = r.baked_color;
                o["baked_normal"] = r.baked_normal;
            }
            if (r.has_metrics) o["metrics"] = metrics_dict(r.metrics);
            return o;
        }, py::arg("target_faces"), py::arg("out_dir"), py::arg("options") = py::none(), py::arg("bake_color") = true,
           py::arg("bake_normal") = true, py::arg("atlas_size") = py::none(), py::arg("crease_angle") = 60.0,
           py::arg("compute_metrics") = true, py::arg("auto_tolerance") = py::none(), py::arg("fast") = false,
           py::arg("progress") = py::none());

    m.def("trimesh_vertex_normals", [](ArrD P, ArrI F) { return arr3(trimesh_vertex_normals(vecd(P), veci(F))); });
    m.def("export_geometry", [](const std::string& path, ArrD P, ArrI F, py::object normals) {
        if (normals.is_none()) return export_geometry(path, vecd(P), veci(F));
        VecD n = vecd(normals.cast<ArrD>());
        return export_geometry(path, vecd(P), veci(F), &n);
    }, py::arg("path"), py::arg("positions"), py::arg("faces"), py::arg("normals") = py::none());
    m.def("add_wireframe", [](py::object glb_in, ArrD P, ArrI F, const std::string& out, std::vector<double> color,
                              double lift, py::object alpha) {
        double c[4] = {color[0], color[1], color[2], color.size() > 3 ? color[3] : 1.0};
        return add_wireframe(glb_in.is_none() ? std::string() : glb_in.cast<std::string>(), vecd(P), veci(F), out, c,
                             lift, alpha.is_none() ? -1.0 : alpha.cast<double>());
    }, py::arg("glb_in"), py::arg("P"), py::arg("F"), py::arg("glb_out"), py::arg("color") = std::vector<double>{0.0, 0.75, 1.0, 1.0},
       py::arg("lift") = 2e-4, py::arg("alpha") = py::none());

    m.attr("DETAIL_LEVELS") = detail_levels();
    // threads: 0 = all cores (FAQEM_THREADS caps it); results are identical for any count
    m.def("set_threads", &set_num_threads, py::arg("n"));
    m.def("num_threads", &num_threads);
    m.def("set_collapse_parallel", &set_collapse_parallel, py::arg("on"),
          "use the threads inside the simplifier's per-collapse check (metrics / bake always do)");
    // "portable" (same bits on macOS / Linux / Windows) or "python-parity" (same bits as faqem/ here)
    m.attr("BUILD_MODE") = std::string(fm::build_mode());

    m.def("load_asset", [](const std::string& path) {
        Asset a;
        {
            py::gil_scoped_release nogil;
            a = load_asset(path);
        }
        return asset_dict(a);
    }, py::arg("path"));

    // ---- compat layer, exposed for the parity tests --------------------------------------------
    py::module_ t = m.def_submodule("_compat");
    t.def("sum", [](ArrD a) { return np::sum(a.data(), (i64)a.size()); });
    t.def("argsort_i64", [](ArrI a) { VecI r = np::argsort_quick(veci(a)); return arr(r, {(py::ssize_t)r.size()}); });
    t.def("argsort_f64", [](ArrD a) { VecI r = np::argsort_quick(vecd(a)); return arr(r, {(py::ssize_t)r.size()}); });
    t.def("rng_random", [](uint64_t seed, int64_t n) { np::Generator g(seed); VecD r = g.random(n); return arr(r, {n}); });
    t.def("rng_permutation", [](uint64_t seed, int64_t n) {
        np::Generator g(seed);
        VecI r = g.permutation(n);
        return arr(r, {n});
    });
    t.def("rng_choice", [](uint64_t seed, int64_t size, ArrD p) {
        np::Generator g(seed);
        VecI r = g.choice((i64)p.size(), size, vecd(p));
        return arr(r, {size});
    });
    t.def("linspace", [](double a, double b, int64_t n) { VecD r = np::linspace(a, b, n); return arr(r, {n}); });
    t.def("find_virtual_edges", [](ArrD P, ArrI F, double tau) {
        VecI r = find_virtual_edges(vecd(P), veci(F), tau);
        return arr(r, {(py::ssize_t)(r.size() / 2), 2});
    }, py::arg("P"), py::arg("F"), py::arg("tau") = 0.01);
    t.def("py_pow", [](double x, double y) { return np::py_pow(x, y); });

    t.def("transform_points", [](ArrD V, ArrD T, bool forder) {
        accel::Mat4 M;
        for (int i = 0; i < 16; i++) M.m[i] = T.data()[i];
        M.forder = forder;
        return arr3(accel::transform_points(vecd(V), M));
    });
    t.def("dot4", [](ArrD A, bool fa, ArrD B, bool fb) {
        accel::Mat4 a, b;
        for (int i = 0; i < 16; i++) { a.m[i] = A.data()[i]; b.m[i] = B.data()[i]; }
        a.forder = fa; b.forder = fb;
        accel::Mat4 c = accel::dot4(a, b);
        return arr(VecD(c.m, c.m + 16), {4, 4});
    });
    t.def("quaternion_matrix", [](ArrD q) {
        accel::Mat4 c = accel::quaternion_matrix(q.data());
        return arr(VecD(c.m, c.m + 16), {4, 4});
    });
    t.def("fix_rigid", [](ArrD T, bool forder) {
        accel::Mat4 M;
        for (int i = 0; i < 16; i++) M.m[i] = T.data()[i];
        M.forder = forder;
        accel::Mat4 c = accel::fix_rigid(M);
        return arr(VecD(c.m, c.m + 16), {4, 4});
    });
    t.def("det3_negative", [](ArrD T) {
        accel::Mat4 M;
        for (int i = 0; i < 16; i++) M.m[i] = T.data()[i];
        return accel::det3_negative(M);
    });
}
