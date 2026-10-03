// Port of faqem/io.py load_asset (+ Asset properties).
#include "io/asset.hpp"

#include <cstdio>
#include <set>
#include <stdexcept>

#include "compat/accelerate.hpp"

namespace faqem {

static double srgb_to_linear(double c) { return c <= 0.04045 ? c / 12.92 : np::py_pow((c + 0.055) / 1.055, 2.4); }

bool Asset::has_texture() const {
    bool any = false;
    for (auto& m : materials) any = any || (bool)m.texture;
    return any && has_uv;
}

bool Asset::has_appearance() const {
    if (has_texture() || has_col) return true;
    std::set<std::vector<double>> cols;
    for (auto& m : materials) {
        std::vector<double> c(4);
        for (int k = 0; k < 4; k++) c[k] = np::rint(m.color[k] * 10000.0) / 10000.0;  // np.round(x, 4)
        cols.insert(c);
    }
    return cols.size() > 1;
}

// Python round(x, 4): correctly rounded decimal
static double pyround4(double x) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.4f", x);
    return std::strtod(buf, nullptr);
}

bool Asset::has_mr_variation() const {
    bool any = false;
    for (auto& m : materials) any = any || (bool)m.mr_texture;
    if (any && has_uv) return true;
    std::set<std::pair<double, double>> s;
    for (auto& m : materials) s.insert({pyround4(m.metallic), pyround4(m.roughness)});
    return s.size() > 1;
}

static std::string lower_ext(const std::string& path) {
    size_t d = path.find_last_of('.');
    std::string e = d == std::string::npos ? "" : path.substr(d);
    for (auto& c : e) c = (char)tolower(c);
    return e;
}

Asset load_asset(const std::string& path) {
    std::string ext = lower_ext(path);
    LoadedScene S;
    if (ext == ".glb" || ext == ".gltf") S = load_gltf_scene(path);
    else if (ext == ".stl") S = load_stl_scene(path);
    else if (ext == ".off") S = load_off_scene(path);
    else if (ext == ".ply") S = load_ply_scene(path);
    else if (ext == ".obj") S = load_obj_scene(path);
    else throw std::runtime_error("unsupported file type for the C++ loader: " + ext);

    Asset A;
    std::vector<int> parts;
    for (size_t i = 0; i < S.nodes.size(); i++)
        if (!S.geometries[S.nodes[i].geometry].faces.empty()) parts.push_back((int)i);
    if (parts.empty()) throw std::runtime_error("No triangle mesh found in the file.");
    std::map<int, int> mat_keys;
    int64_t off = 0;
    // per-part corner attributes (empty = None)
    std::vector<VecD> uvs, cols;
    for (int pi : parts) {
        const SceneNode& node = S.nodes[pi];
        const Geometry& g = S.geometries[node.geometry];
        accel::Mat4 T;
        std::memcpy(T.m, node.T, sizeof(T.m));
        T.forder = node.forder;
        VecD V = accel::transform_points(g.vertices, T);
        int64_t nv = (int64_t)V.size() / 3, nf = (int64_t)g.faces.size() / 3;
        VecI Fc = g.faces;
        if (accel::det3_negative(T))
            for (int64_t f = 0; f < nf; f++) std::swap(Fc[f * 3], Fc[f * 3 + 2]);
        const Visual& vis = g.visual;
        int key = vis.texture_visuals ? vis.material_key : -1;
        if (!mat_keys.count(key)) {
            mat_keys[key] = (int)A.materials.size();
            A.materials.push_back(key >= 0 ? S.materials.at(key) : Material());
        }
        int mid = mat_keys[key];
        if (vis.has_uv && (int64_t)vis.uv.size() / 2 == nv) {
            VecD u(nf * 6);
            for (int64_t f = 0; f < nf; f++)
                for (int k = 0; k < 3; k++)
                    for (int c = 0; c < 2; c++) u[(f * 3 + k) * 2 + c] = vis.uv[Fc[f * 3 + k] * 2 + c];
            uvs.push_back(std::move(u));
            A.has_uv = true;
        } else {
            uvs.emplace_back();
        }
        bool appended = false;
        if (vis.color_kind == 2) {
            // ColorVisuals face colours (uint8 RGBA per face)
            VecD c(nf * 12);
            for (int64_t f = 0; f < nf; f++)
                for (int k = 0; k < 3; k++) {
                    for (int j = 0; j < 3; j++) c[(f * 3 + k) * 4 + j] = srgb_to_linear(vis.colors[f * 4 + j] / 255.0);
                    c[(f * 3 + k) * 4 + 3] = vis.colors[f * 4 + 3] / 255.0;
                }
            cols.push_back(std::move(c));
            A.has_col = true;
            appended = true;
        } else if ((vis.color_kind == 1 || vis.color_kind == 3) && (int64_t)vis.colors.size() / vis.color_cols == nv) {
            int cc = vis.color_cols;
            VecD c(nf * 3 * cc);
            double mx = -INFINITY;
            for (int64_t f = 0; f < nf; f++)
                for (int k = 0; k < 3; k++)
                    for (int j = 0; j < cc; j++) {
                        double v = vis.colors[Fc[f * 3 + k] * cc + j];
                        c[(f * 3 + k) * cc + j] = v;
                        mx = np::fmax0(mx, v);
                    }
            if (mx > 1.0)
                for (auto& v : c) v = v / 255.0;
            VecD o(nf * 12);
            for (int64_t i = 0; i < nf * 3; i++) {
                for (int j = 0; j < 3; j++) o[i * 4 + j] = srgb_to_linear(c[i * cc + j]);
                o[i * 4 + 3] = cc >= 4 ? c[i * cc + 3] : 1.0;
            }
            cols.push_back(std::move(o));
            A.has_col = true;
            appended = true;
        }
        if (!appended) cols.emplace_back();
        A.positions.insert(A.positions.end(), V.begin(), V.end());
        for (int64_t i = 0; i < nf * 3; i++) A.faces.push_back(Fc[i] + off);
        for (int64_t f = 0; f < nf; f++) A.face_material.push_back(mid);
        off += nv;
    }
    int64_t m = (int64_t)A.faces.size() / 3;
    if (A.has_uv) {
        A.corner_uv.assign(m * 6, 0.0);
        int64_t o = 0;
        for (size_t i = 0; i < parts.size(); i++) {
            int64_t k = (int64_t)S.geometries[S.nodes[parts[i]].geometry].faces.size() / 3;
            if (!uvs[i].empty()) std::copy(uvs[i].begin(), uvs[i].end(), A.corner_uv.begin() + o * 6);
            o += k;
        }
    }
    if (A.has_col) {
        A.corner_color.assign(m * 12, 1.0);
        int64_t o = 0;
        for (size_t i = 0; i < parts.size(); i++) {
            int64_t k = (int64_t)S.geometries[S.nodes[parts[i]].geometry].faces.size() / 3;
            if (!cols[i].empty()) std::copy(cols[i].begin(), cols[i].end(), A.corner_color.begin() + o * 12);
            o += k;
        }
    }
    A.info = {{"parts", (double)parts.size()},
              {"materials", (double)A.materials.size()},
              {"textured", A.has_texture() ? 1.0 : 0.0},
              {"vertex_colors", A.has_col ? 1.0 : 0.0}};
    return A;
}

}  // namespace faqem
