#include "io/export.hpp"

#include <charconv>
#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "compat/accelerate.hpp"
#include "compat/fmath.hpp"
#include "io/asset.hpp"

namespace faqem {

using ojson = nlohmann::ordered_json;

// ------------------------------------------------------------------------------------------------
// trimesh vertex normals: face normals (unitize of cross(t1 - t0, t2 - t1)), corner angles,
// angle-weighted sum through scipy's coo_matvecs (fused multiply-add), unitize
// ------------------------------------------------------------------------------------------------
VecD trimesh_vertex_normals(const VecD& P, const VecI& F) {
    int64_t nV = (int64_t)P.size() / 3, m = (int64_t)F.size() / 3;
    VecD crosses(m * 3), e01(m * 3), e02(m * 3), e12(m * 3);
    for (int64_t f = 0; f < m; f++) {
        const double* t0 = &P[F[f * 3] * 3];
        const double* t1 = &P[F[f * 3 + 1] * 3];
        const double* t2 = &P[F[f * 3 + 2] * 3];
        double a[3], b[3];
        for (int k = 0; k < 3; k++) {
            a[k] = t1[k] - t0[k];
            b[k] = t2[k] - t1[k];
            e01[f * 3 + k] = t1[k] - t0[k];
            e02[f * 3 + k] = t2[k] - t0[k];
            e12[f * 3 + k] = t2[k] - t1[k];
        }
        np::cross(a[0], a[1], a[2], b[0], b[1], b[2], crosses[f * 3], crosses[f * 3 + 1], crosses[f * 3 + 2]);
    }
    std::vector<char> valid;
    VecD un = accel::unitize_rows(crosses, &valid);
    VecD fn(m * 3, 0.0);
    for (int64_t f = 0; f < m; f++)
        if (valid[f])
            for (int k = 0; k < 3; k++) fn[f * 3 + k] = un[f * 3 + k];
    // triangles.angles
    VecD u = accel::unitize_rows(e01), v = accel::unitize_rows(e02), w = accel::unitize_rows(e12);
    VecD uv(m * 3), uw(m * 3);
    for (int64_t i = 0; i < m * 3; i++) {
        uv[i] = u[i] * v[i];
        uw[i] = (-u[i]) * w[i];
    }
    VecD duv = accel::rowdot_ones(uv, m, 3), duw = accel::rowdot_ones(uw, m, 3);
    VecD ang(m * 3);
    for (int64_t f = 0; f < m; f++) {
        double a0 = fm::acos(np::clip(duv[f], -1, 1));
        double a1 = fm::acos(np::clip(duw[f], -1, 1));
        double a2 = PI - a0 - a1;
        if (a0 < 1e-8 || a1 < 1e-8 || a2 < 1e-8) a0 = a1 = a2 = 0.0;
        ang[f * 3] = a0;
        ang[f * 3 + 1] = a1;
        ang[f * 3 + 2] = a2;
    }
    // face_ok = (fn ** 2).sum(axis=1) > 0.5; coo_matvecs over (vertex, face) entries
    VecD Y(nV * 3, 0.0);
    for (int64_t f = 0; f < m; f++) {
        double s = (fn[f * 3] * fn[f * 3] + fn[f * 3 + 1] * fn[f * 3 + 1]) + fn[f * 3 + 2] * fn[f * 3 + 2];
        if (!(s > 0.5)) continue;
        for (int k = 0; k < 3; k++) {
            int64_t r = F[f * 3 + k];
            double d = ang[f * 3 + k];
            for (int j = 0; j < 3; j++) Y[r * 3 + j] = std::fma(d, fn[f * 3 + j], Y[r * 3 + j]);
        }
    }
    return accel::unitize_rows(Y);
}

// ------------------------------------------------------------------------------------------------
// minimal GLB writer (structure follows trimesh's exporter)
// ------------------------------------------------------------------------------------------------
namespace {

struct GlbBuilder {
    ojson tree;
    std::vector<uint8_t> bin;
    GlbBuilder() {
        tree["scene"] = 0;
        tree["scenes"] = ojson::array({ojson{{"nodes", ojson::array({0})}}});
        tree["asset"] = ojson{{"version", "2.0"}, {"generator", "faqem-cpp"}};
        tree["accessors"] = ojson::array();
        tree["meshes"] = ojson::array();
        tree["materials"] = ojson::array();
        tree["textures"] = ojson::array();
        tree["images"] = ojson::array();
        tree["bufferViews"] = ojson::array();
    }
    int view(const void* data, size_t n, int target = -1) {
        while (bin.size() % 4) bin.push_back(0);
        size_t start = bin.size();
        bin.insert(bin.end(), (const uint8_t*)data, (const uint8_t*)data + n);
        ojson v{{"buffer", 0}, {"byteOffset", start}, {"byteLength", n}};
        if (target > 0) v["target"] = target;
        tree["bufferViews"].push_back(v);
        return (int)tree["bufferViews"].size() - 1;
    }
    int accessor_f32(const std::vector<float>& a, int comps, const char* type, bool minmax) {
        int bv = view(a.data(), a.size() * 4, 34962);
        ojson acc{{"bufferView", bv}, {"componentType", 5126}, {"count", a.size() / comps}, {"type", type}};
        if (minmax && !a.empty()) {
            std::vector<double> mn(comps, INFINITY), mx(comps, -INFINITY);
            for (size_t i = 0; i < a.size(); i++) {
                mn[i % comps] = std::min(mn[i % comps], (double)a[i]);
                mx[i % comps] = std::max(mx[i % comps], (double)a[i]);
            }
            acc["min"] = mn;
            acc["max"] = mx;
        }
        tree["accessors"].push_back(acc);
        return (int)tree["accessors"].size() - 1;
    }
    int accessor_u32(const std::vector<uint32_t>& a) {
        int bv = view(a.data(), a.size() * 4, 34963);
        tree["accessors"].push_back(ojson{{"bufferView", bv}, {"componentType", 5125}, {"count", a.size()}, {"type", "SCALAR"}});
        return (int)tree["accessors"].size() - 1;
    }
    int image(const Image& img) {
        std::vector<uint8_t> png = encode_png(img);
        int bv = view(png.data(), png.size());
        tree["images"].push_back(ojson{{"bufferView", bv}, {"mimeType", "image/png"}});
        tree["textures"].push_back(ojson{{"source", (int)tree["images"].size() - 1}});
        return (int)tree["textures"].size() - 1;
    }
    void write(const std::string& path) {
        tree["nodes"] = ojson::array({ojson{{"name", "geometry_0"}, {"mesh", 0}}});
        while (bin.size() % 4) bin.push_back(0);
        tree["buffers"] = ojson::array({ojson{{"byteLength", bin.size()}}});
        for (const char* k : {"materials", "textures", "images"})
            if (tree[k].empty()) tree.erase(k);
        std::string js = tree.dump();
        while (js.size() % 4) js.push_back(' ');
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot write " + path);
        auto u32 = [&](uint32_t v) { f.write((const char*)&v, 4); };
        u32(0x46546C67);
        u32(2);
        u32((uint32_t)(12 + 8 + js.size() + 8 + bin.size()));
        u32((uint32_t)js.size());
        u32(0x4E4F534A);
        f.write(js.data(), js.size());
        u32((uint32_t)bin.size());
        u32(0x004E4942);
        f.write((const char*)bin.data(), bin.size());
    }
};

std::vector<float> to_f32(const VecD& a) {
    std::vector<float> o(a.size());
    for (size_t i = 0; i < a.size(); i++) o[i] = (float)a[i];
    return o;
}
std::vector<uint32_t> to_u32(const VecI& a) {
    std::vector<uint32_t> o(a.size());
    for (size_t i = 0; i < a.size(); i++) o[i] = (uint32_t)a[i];
    return o;
}
std::vector<double> color_factor(const double c[4]) {
    std::vector<double> o(4);
    for (int k = 0; k < 4; k++) o[k] = (double)(uint8_t)np::rint(c[k] * 255) / 255.0;
    return o;
}
bool ends_with(const std::string& s, const char* e) {
    size_t n = std::strlen(e);
    return s.size() >= n && s.compare(s.size() - n, n, e) == 0;
}
std::string lowercase(std::string s) {
    for (auto& c : s) c = (char)tolower(c);
    return s;
}

}  // namespace

std::string export_geometry(const std::string& path, const VecD& positions, const VecI& faces, const VecD* normals,
                            const double color_in[4]) {
    double color[4] = {0.55, 0.56, 0.6, 1.0};
    if (color_in)
        for (int k = 0; k < 4; k++) color[k] = color_in[k];
    std::string lp = lowercase(path);
    if (ends_with(lp, ".glb") || ends_with(lp, ".gltf")) {
        GlbBuilder G;
        int idx = G.accessor_u32(to_u32(faces));
        int pos = G.accessor_f32(to_f32(positions), 3, "VEC3", true);
        ojson prim{{"attributes", ojson{{"POSITION", pos}}}, {"indices", idx}, {"mode", 4}};
        ojson mat{{"pbrMetallicRoughness", ojson{{"baseColorFactor", color_factor(color)},
                                                  {"roughnessFactor", 0.8},
                                                  {"metallicFactor", 0.0}}},
                  {"doubleSided", true}};
        G.tree["materials"].push_back(mat);
        prim["material"] = 0;
        if (normals) prim["attributes"]["NORMAL"] = G.accessor_f32(to_f32(accel::unitize_rows(*normals)), 3, "VEC3", false);
        G.tree["meshes"].push_back(ojson{{"name", "geometry_0"}, {"primitives", ojson::array({prim})}});
        G.write(path);
        return path;
    }
    // .obj: trimesh export_obj text (no normals: a fresh Trimesh has none cached)
    std::string out = "# https://github.com/mikedh/trimesh\n";
    char buf[128];
    int64_t nv = (int64_t)positions.size() / 3, nf = (int64_t)faces.size() / 3;
    std::string body;
    body.reserve((size_t)nv * 40 + (size_t)nf * 24);
    // std::to_chars: exactly rounded "%.8f" / "%d" by the C++17 spec, independent of the C library
    auto put_f = [&](double x) {
        auto r = std::to_chars(buf, buf + sizeof buf, x, std::chars_format::fixed, 8);
        body.append(buf, r.ptr);
    };
    auto put_i = [&](long long x) {
        auto r = std::to_chars(buf, buf + sizeof buf, x);
        body.append(buf, r.ptr);
    };
    for (int64_t i = 0; i < nv; i++) {
        body += i ? "\nv " : "v ";
        for (int k = 0; k < 3; k++) {
            if (k) body += ' ';
            put_f(positions[i * 3 + k]);
        }
    }
    for (int64_t f = 0; f < nf; f++) {
        body += "\nf ";
        for (int k = 0; k < 3; k++) {
            if (k) body += ' ';
            put_i((long long)faces[f * 3 + k] + 1);
        }
    }
    out += body + "\n\n";
    std::ofstream fo(path, std::ios::binary);
    if (!fo) throw std::runtime_error("cannot write " + path);
    fo << out;
    return path;
}

std::string export_textured(const std::string& path, const VecD& positions, const VecD& normals, const VecD& uv,
                            const Image* color_img, const Image* normal_img, const Image* mr_img,
                            const double base_color[4], double metallic, double roughness, bool double_sided,
                            const std::string& alpha_mode) {
    int64_t n = (int64_t)positions.size() / 3;
    VecI faces(n);
    for (int64_t i = 0; i < n; i++) faces[i] = i;
    GlbBuilder G;
    int idx = G.accessor_u32(to_u32(faces));
    int pos = G.accessor_f32(to_f32(positions), 3, "VEC3", true);
    ojson prim{{"attributes", ojson{{"POSITION", pos}}}, {"indices", idx}, {"mode", 4}};
    // material (baseColorTexture, normalTexture, metallicRoughnessTexture images in trimesh's order)
    ojson pbr;
    ojson mat;
    double white[4] = {1, 1, 1, 1};
    std::vector<double> bcf = color_factor(color_img ? white : base_color);
    int tex_color = color_img ? G.image(*color_img) : -1;
    int tex_normal = normal_img ? G.image(*normal_img) : -1;
    int tex_mr = mr_img ? G.image(*mr_img) : -1;
    if (tex_color >= 0) pbr["baseColorTexture"] = ojson{{"index", tex_color}};
    pbr["baseColorFactor"] = bcf;
    pbr["roughnessFactor"] = mr_img ? 1.0 : roughness;
    pbr["metallicFactor"] = mr_img ? 1.0 : metallic;
    if (tex_mr >= 0) pbr["metallicRoughnessTexture"] = ojson{{"index", tex_mr}};
    mat["pbrMetallicRoughness"] = pbr;
    mat["name"] = "faqem_baked";
    mat["alphaMode"] = (alpha_mode == "OPAQUE" || alpha_mode == "MASK" || alpha_mode == "BLEND") ? alpha_mode : "OPAQUE";
    mat["doubleSided"] = double_sided;
    if (tex_normal >= 0) mat["normalTexture"] = ojson{{"index", tex_normal}};
    G.tree["materials"].push_back(mat);
    // uv: trimesh stores v flipped back (1 - v) and casts to float32
    VecD tuv(uv.size());
    for (int64_t i = 0; i < n; i++) {
        tuv[i * 2] = uv[i * 2];
        tuv[i * 2 + 1] = 1.0 - uv[i * 2 + 1];
    }
    prim["attributes"]["TEXCOORD_0"] = G.accessor_f32(to_f32(tuv), 2, "VEC2", false);
    prim["material"] = 0;
    prim["attributes"]["NORMAL"] = G.accessor_f32(to_f32(accel::unitize_rows(normals)), 3, "VEC3", false);
    G.tree["meshes"].push_back(ojson{{"name", "geometry_0"}, {"primitives", ojson::array({prim})}});
    G.write(path);
    return path;
}

// ------------------------------------------------------------------------------------------------
// wireframe overlays (port of faqem/wireframe.py)
// ------------------------------------------------------------------------------------------------
std::string add_wireframe(const std::string& glb_in, const VecD& P, const VecI& F, const std::string& glb_out,
                          const double color[4], double lift, double alpha) {
    ojson js;
    std::vector<uint8_t> binary;
    if (glb_in.empty()) {
        js = ojson{{"asset", {{"version", "2.0"}, {"generator", "faqem"}}},
                   {"scene", 0},
                   {"scenes", ojson::array({ojson{{"nodes", ojson::array()}}})},
                   {"buffers", ojson::array({ojson{{"byteLength", 0}}})}};
        lift = 0.0;
    } else {
        std::vector<uint8_t> d = read_file(glb_in);
        uint32_t magic;
        std::memcpy(&magic, d.data(), 4);
        if (magic != 0x46546C67) throw std::runtime_error("not a GLB file");
        size_t off = 12;
        while (off + 8 <= d.size()) {
            uint32_t ln, typ;
            std::memcpy(&ln, &d[off], 4);
            std::memcpy(&typ, &d[off + 4], 4);
            if (typ == 0x4E4F534A) js = ojson::parse(d.begin() + off + 8, d.begin() + off + 8 + ln);
            else if (typ == 0x004E4942) binary.assign(d.begin() + off + 8, d.begin() + off + 8 + ln);
            off += 8 + ln;
        }
    }
    int64_t nV = (int64_t)P.size() / 3, m = (int64_t)F.size() / 3;
    // area-weighted vertex normals for the lift (np.add.at, face order per corner)
    VecD vn(nV * 3, 0.0);
    VecD n(m * 3);
    for (int64_t f = 0; f < m; f++) {
        const double* a = &P[F[f * 3] * 3];
        const double* b = &P[F[f * 3 + 1] * 3];
        const double* c = &P[F[f * 3 + 2] * 3];
        np::cross(b[0] - a[0], b[1] - a[1], b[2] - a[2], c[0] - a[0], c[1] - a[1], c[2] - a[2], n[f * 3], n[f * 3 + 1], n[f * 3 + 2]);
    }
    for (int k = 0; k < 3; k++)
        for (int64_t f = 0; f < m; f++)
            for (int j = 0; j < 3; j++) vn[F[f * 3 + k] * 3 + j] += n[f * 3 + j];
    for (int64_t i = 0; i < nV; i++) {
        double l = np::fmax0(np::norm3(vn[i * 3], vn[i * 3 + 1], vn[i * 3 + 2]), 1e-30);
        for (int j = 0; j < 3; j++) vn[i * 3 + j] /= l;
    }
    double lo[3], hi[3];
    for (int k = 0; k < 3; k++) {
        lo[k] = hi[k] = P[k];
        for (int64_t i = 1; i < nV; i++) {
            lo[k] = np::fmin0(lo[k], P[i * 3 + k]);
            hi[k] = np::fmax0(hi[k], P[i * 3 + k]);
        }
    }
    double diag = np::norm3(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    if (diag == 0) diag = 1.0;
    std::vector<float> V(nV * 3);
    double s = lift * diag;
    for (int64_t i = 0; i < nV * 3; i++) V[i] = (float)(P[i] + vn[i] * s);
    // unique sorted edges
    VecI key(3 * m);
    for (int k = 0; k < 3; k++)
        for (int64_t f = 0; f < m; f++) {
            int64_t a = F[f * 3 + k], b = F[f * 3 + (k + 1) % 3];
            key[k * m + f] = std::min(a, b) * nV + std::max(a, b);
        }
    std::sort(key.begin(), key.end());
    key.erase(std::unique(key.begin(), key.end()), key.end());
    std::vector<uint32_t> E(key.size() * 2);
    for (size_t i = 0; i < key.size(); i++) {
        E[i * 2] = (uint32_t)(key[i] / nV);
        E[i * 2 + 1] = (uint32_t)(key[i] % nV);
    }
    if (alpha < 0) {
        // auto_alpha: median edge length vs model size, viewer of 650 px
        VecD el(key.size());
        for (size_t i = 0; i < key.size(); i++) {
            const double* a = &P[E[i * 2] * 3];
            const double* b = &P[E[i * 2 + 1] * 3];
            el[i] = np::norm3(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        }
        double med = 0;
        if (!el.empty()) {
            size_t h = el.size() / 2;
            std::nth_element(el.begin(), el.begin() + h, el.end());
            double upper = el[h];
            if (el.size() % 2) med = upper;
            else {
                double lower = *std::max_element(el.begin(), el.begin() + h);
                med = (lower + upper) / 2;
            }
        }
        double spacing_px = med / diag * 650.0;
        alpha = np::clip(spacing_px / 6.0, 0.12, 1.0);
    }
    while (binary.size() % 4) binary.push_back(0);
    if (!js.contains("buffers")) js["buffers"] = ojson::array({ojson{{"byteLength", 0}}});
    if (!js.contains("bufferViews")) js["bufferViews"] = ojson::array();
    if (!js.contains("accessors")) js["accessors"] = ojson::array();
    auto append = [&](const void* data, size_t nbytes, int target) {
        size_t start = binary.size();
        binary.insert(binary.end(), (const uint8_t*)data, (const uint8_t*)data + nbytes);
        while (binary.size() % 4) binary.push_back(0);
        js["bufferViews"].push_back(ojson{{"buffer", 0}, {"byteOffset", start}, {"byteLength", nbytes}, {"target", target}});
        return (int)js["bufferViews"].size() - 1;
    };
    int pv = append(V.data(), V.size() * 4, 34962);
    std::vector<double> vmin(3, INFINITY), vmax(3, -INFINITY);
    for (int64_t i = 0; i < nV * 3; i++) {
        vmin[i % 3] = std::min(vmin[i % 3], (double)V[i]);
        vmax[i % 3] = std::max(vmax[i % 3], (double)V[i]);
    }
    js["accessors"].push_back(ojson{{"bufferView", pv}, {"componentType", 5126}, {"count", nV}, {"type", "VEC3"}, {"min", vmin}, {"max", vmax}});
    int pos_acc = (int)js["accessors"].size() - 1;
    int iv = append(E.data(), E.size() * 4, 34963);
    js["accessors"].push_back(ojson{{"bufferView", iv}, {"componentType", 5125}, {"count", E.size()}, {"type", "SCALAR"}});
    int idx_acc = (int)js["accessors"].size() - 1;
    if (!js.contains("materials")) js["materials"] = ojson::array();
    ojson mat{{"name", "wireframe"},
              {"pbrMetallicRoughness", {{"baseColorFactor", {color[0], color[1], color[2], alpha}}, {"metallicFactor", 0.0}, {"roughnessFactor", 1.0}}},
              {"extensions", {{"KHR_materials_unlit", ojson::object()}}}};
    if (alpha < 1.0) mat["alphaMode"] = "BLEND";
    js["materials"].push_back(mat);
    if (!js.contains("extensionsUsed")) js["extensionsUsed"] = ojson::array();
    bool has = false;
    for (auto& e : js["extensionsUsed"]) has = has || e == "KHR_materials_unlit";
    if (!has) js["extensionsUsed"].push_back("KHR_materials_unlit");
    if (!js.contains("meshes")) js["meshes"] = ojson::array();
    js["meshes"].push_back(ojson{{"name", "wireframe"},
                                 {"primitives", ojson::array({ojson{{"attributes", {{"POSITION", pos_acc}}}, {"indices", idx_acc}, {"mode", 1}, {"material", (int)js["materials"].size() - 1}}})}});
    if (!js.contains("nodes")) js["nodes"] = ojson::array();
    js["nodes"].push_back(ojson{{"name", "wireframe"}, {"mesh", (int)js["meshes"].size() - 1}});
    if (!js.contains("scenes")) js["scenes"] = ojson::array({ojson{{"nodes", ojson::array()}}});
    int sc = js.value("scene", 0);
    js["scenes"][sc]["nodes"].push_back((int)js["nodes"].size() - 1);
    js["buffers"][0]["byteLength"] = binary.size();
    js["buffers"][0].erase("uri");
    std::string jb = js.dump();
    while (jb.size() % 4) jb.push_back(' ');
    std::ofstream f(glb_out, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + glb_out);
    auto u32 = [&](uint32_t v) { f.write((const char*)&v, 4); };
    u32(0x46546C67);
    u32(2);
    u32((uint32_t)(12 + 8 + jb.size() + 8 + binary.size()));
    u32((uint32_t)jb.size());
    u32(0x4E4F534A);
    f.write(jb.data(), jb.size());
    u32((uint32_t)binary.size());
    u32(0x004E4942);
    f.write((const char*)binary.data(), binary.size());
    return glb_out;
}

}  // namespace faqem
