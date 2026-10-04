// glTF 2.0 / GLB loading with trimesh 5.1.0 semantics (trimesh.exchange.gltf._read_buffers +
// trimesh.scene.transforms), as used by trimesh.load(path, force="scene", process=False).
#include <draco/compression/decode.h>

#include <deque>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>

#include "compat/accelerate.hpp"
#include "io/asset.hpp"

namespace faqem {

using json = nlohmann::json;

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

namespace {

std::string dirname(const std::string& p) {
    size_t s = p.find_last_of("/\\");
    return s == std::string::npos ? std::string(".") : p.substr(0, s);
}

std::vector<uint8_t> base64_decode(const std::string& s) {
    static int T[256];
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 256; i++) T[i] = -1;
        const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) T[(unsigned char)a[i]] = i;
        T[(unsigned char)'-'] = 62;
        T[(unsigned char)'_'] = 63;
        init = true;
    }
    std::vector<uint8_t> out;
    int val = 0, bits = -8;
    for (unsigned char c : s) {
        if (T[c] < 0) continue;
        val = (val << 6) + T[c];
        bits += 6;
        if (bits >= 0) {
            out.push_back((uint8_t)((val >> bits) & 0xff));
            bits -= 8;
        }
    }
    return out;
}

std::string url_unquote(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '%' && i + 2 < s.size()) {
            o.push_back((char)std::stoi(s.substr(i + 1, 2), nullptr, 16));
            i += 2;
        } else {
            o.push_back(s[i]);
        }
    }
    return o;
}

std::vector<uint8_t> uri_to_bytes(const std::string& uri, const std::string& dir) {
    if (uri.rfind("data:", 0) == 0) {
        size_t c = uri.find(',');
        return base64_decode(uri.substr(c + 1));
    }
    try {
        return read_file(dir + "/" + uri);
    } catch (...) {
        return read_file(dir + "/" + url_unquote(uri));
    }
}

// numpy dtype per glTF componentType
int itemsize(int ct) {
    switch (ct) {
        case 5120: case 5121: return 1;
        case 5122: case 5123: return 2;
        case 5125: case 5126: return 4;
    }
    throw std::runtime_error("unsupported glTF componentType");
}
int ncomp(const std::string& t) {
    if (t == "SCALAR") return 1;
    if (t == "VEC2") return 2;
    if (t == "VEC3") return 3;
    if (t == "VEC4") return 4;
    if (t == "MAT2") return 4;
    if (t == "MAT3") return 9;
    if (t == "MAT4") return 16;
    throw std::runtime_error("unsupported glTF accessor type " + t);
}

struct Accessor {
    int ct = 5126;          // componentType
    int nc = 1;             // components per item
    int64_t count = 0;
    std::vector<uint8_t> bytes;  // contiguous count * nc * itemsize
    template <typename T>
    T raw(int64_t i) const {
        T v;
        std::memcpy(&v, &bytes[i * sizeof(T)], sizeof(T));
        return v;
    }
    // element as float64 (numpy astype(float64) of the stored dtype)
    double get(int64_t i) const {
        switch (ct) {
            case 5120: return (double)raw<int8_t>(i);
            case 5121: return (double)raw<uint8_t>(i);
            case 5122: return (double)raw<int16_t>(i);
            case 5123: return (double)raw<uint16_t>(i);
            case 5125: return (double)raw<uint32_t>(i);
            default: return (double)raw<float>(i);
        }
    }
    int64_t geti(int64_t i) const {
        switch (ct) {
            case 5120: return raw<int8_t>(i);
            case 5121: return raw<uint8_t>(i);
            case 5122: return raw<int16_t>(i);
            case 5123: return raw<uint16_t>(i);
            case 5125: return raw<uint32_t>(i);
            default: return (int64_t)raw<float>(i);
        }
    }
    int64_t size() const { return count * nc; }
};

void draco_decode(const std::vector<uint8_t>& blob, const json& ext, const json& prim,
                  std::vector<Accessor>& access) {
    draco::DecoderBuffer buf;
    buf.Init((const char*)blob.data(), blob.size());
    draco::Decoder dec;
    auto st = dec.DecodeMeshFromBuffer(&buf);
    if (!st.ok()) throw std::runtime_error("draco decode failed: " + st.status().error_msg_string());
    std::unique_ptr<draco::Mesh> mesh = std::move(st).value();
    std::map<int, std::string> names;
    for (auto& kv : ext["attributes"].items()) names[kv.value().get<int>()] = kv.key();
    const json& attributes = prim.value("attributes", json::object());
    int64_t np_ = mesh->num_points();
    for (int ai = 0; ai < mesh->num_attributes(); ai++) {
        const draco::PointAttribute* a = mesh->attribute(ai);
        auto it = names.find((int)a->unique_id());
        if (it == names.end() || !attributes.contains(it->second)) continue;
        Accessor& acc = access[attributes[it->second].get<int>()];
        int nc = a->num_components();
        acc.nc = nc;
        acc.count = np_;
        if (a->data_type() == draco::DT_FLOAT32) {
            acc.ct = 5126;
            acc.bytes.resize(np_ * nc * 4);
            std::vector<float> v(nc);
            for (int64_t i = 0; i < np_; i++) {
                a->ConvertValue<float>(a->mapped_index(draco::PointIndex((uint32_t)i)), (int8_t)nc, v.data());
                std::memcpy(&acc.bytes[i * nc * 4], v.data(), nc * 4);
            }
        } else if (a->data_type() == draco::DT_UINT8) {
            acc.ct = 5121;
            acc.bytes.resize(np_ * nc);
            std::vector<uint8_t> v(nc);
            for (int64_t i = 0; i < np_; i++) {
                a->ConvertValue<uint8_t>(a->mapped_index(draco::PointIndex((uint32_t)i)), (int8_t)nc, v.data());
                std::memcpy(&acc.bytes[i * nc], v.data(), nc);
            }
        } else if (a->data_type() == draco::DT_UINT16) {
            acc.ct = 5123;
            acc.bytes.resize(np_ * nc * 2);
            std::vector<uint16_t> v(nc);
            for (int64_t i = 0; i < np_; i++) {
                a->ConvertValue<uint16_t>(a->mapped_index(draco::PointIndex((uint32_t)i)), (int8_t)nc, v.data());
                std::memcpy(&acc.bytes[i * nc * 2], v.data(), nc * 2);
            }
        } else {
            throw std::runtime_error("unsupported draco attribute data type");
        }
    }
    if (prim.contains("indices")) {
        Accessor& acc = access[prim["indices"].get<int>()];
        int64_t nf = mesh->num_faces();
        acc.ct = 5125;
        acc.nc = 3;
        acc.count = nf;
        acc.bytes.resize(nf * 12);
        for (int64_t f = 0; f < nf; f++) {
            const auto& face = mesh->face(draco::FaceIndex((uint32_t)f));
            for (int k = 0; k < 3; k++) {
                uint32_t v = face[k].value();
                std::memcpy(&acc.bytes[(f * 3 + k) * 4], &v, 4);
            }
        }
    }
}

// PBRMaterial(**pbr) -> io._material_from_visual
Material parse_material(const json& mat, const json& header, std::vector<ImagePtr>& images,
                        const std::vector<std::vector<uint8_t>>& image_blobs, std::vector<char>& decoded) {
    json loop = mat;
    if (loop.contains("pbrMetallicRoughness")) {
        json pbr = loop["pbrMetallicRoughness"];
        loop.erase("pbrMetallicRoughness");
        for (auto& kv : pbr.items()) loop[kv.key()] = kv.value();
    }
    if (mat.contains("extensions") && mat["extensions"].contains("KHR_materials_pbrSpecularGlossiness"))
        throw std::runtime_error("unsupported: KHR_materials_pbrSpecularGlossiness material");
    auto texture_of = [&](const char* key) -> ImagePtr {
        if (!loop.contains(key) || !loop[key].is_object() || !loop[key].contains("index")) return nullptr;
        const json& tex = header["textures"][loop[key]["index"].get<int>()];
        int src = -1;
        if (tex.contains("extensions") && tex["extensions"].contains("EXT_texture_webp") &&
            tex["extensions"]["EXT_texture_webp"].contains("source"))
            src = tex["extensions"]["EXT_texture_webp"]["source"].get<int>();
        if (src < 0 && tex.contains("source")) src = tex["source"].get<int>();
        if (src < 0 || src >= (int)images.size()) return nullptr;
        if (!decoded[src]) {
            decoded[src] = 1;
            if (!image_blobs[src].empty()) images[src] = decode_image_rgba(image_blobs[src].data(), image_blobs[src].size());
        }
        return images[src];
    };
    Material m;
    if (loop.contains("baseColorFactor") && loop["baseColorFactor"].is_array()) {
        std::vector<double> c;
        for (auto& v : loop["baseColorFactor"]) c.push_back(v.get<double>());
        // to_rgba (float64): uint8(round(clip(c * 255, 0, 255)))
        for (int k = 0; k < 4; k++) {
            double u8 = k < (int)c.size() ? np::rint(np::clip(c[k] * 255, 0.0, 255.0)) : 255.0;
            m.color[k] = (double)(uint8_t)u8 / 255.0;
        }
    } else {
        for (int k = 0; k < 4; k++) m.color[k] = 1.0;
    }
    m.texture = texture_of("baseColorTexture");
    m.metallic = loop.contains("metallicFactor") && !loop["metallicFactor"].is_null() ? loop["metallicFactor"].get<double>() : 0.0;
    m.roughness = loop.contains("roughnessFactor") && !loop["roughnessFactor"].is_null() ? loop["roughnessFactor"].get<double>() : 1.0;
    m.mr_texture = texture_of("metallicRoughnessTexture");
    if (loop.contains("alphaMode") && loop["alphaMode"].is_string()) {
        std::string a = loop["alphaMode"].get<std::string>();
        for (auto& ch : a) ch = (char)toupper(ch);
        a.erase(0, a.find_first_not_of(" \t\n"));
        a.erase(a.find_last_not_of(" \t\n") + 1);
        if (a != "OPAQUE" && a != "MASK" && a != "BLEND") throw std::runtime_error("incorrect alphaMode: " + a);
        m.alpha_mode = a;
    }
    if (loop.contains("doubleSided")) {
        const json& d = loop["doubleSided"];
        m.double_sided = d.is_boolean() ? d.get<bool>() : (d.is_number() ? d.get<double>() != 0 : !d.is_null());
    }
    return m;
}

}  // namespace

VecD to_rgba_f32(const std::vector<float>& c, int cols) {
    size_t n = c.size() / cols;
    VecD out(n * 4, 255.0);
    for (size_t i = 0; i < n; i++)
        for (int k = 0; k < cols && k < 4; k++) {
            float v = c[i * cols + k];
            if (!std::isfinite(v)) v = 0.0f;
            float s = v * 255.0f;
            s = s < 0.0f ? 0.0f : (s > 255.0f ? 255.0f : s);
            s = std::nearbyint(s);
            out[i * 4 + k] = (double)(uint8_t)s;
        }
    return out;
}

LoadedScene load_gltf_scene(const std::string& path) {
    std::vector<uint8_t> file = read_file(path);
    std::string dir = dirname(path);
    json header;
    std::vector<std::vector<uint8_t>> buffers;
    bool glb = file.size() >= 12 && std::memcmp(file.data(), "glTF", 4) == 0;
    if (glb) {
        uint32_t head[5];
        std::memcpy(head, file.data(), 20);
        if (head[1] != 2) throw std::runtime_error("only GLTF 2 is supported");
        uint32_t length = head[2], chunk_length = head[3], chunk_type = head[4];
        if (chunk_type != 0x4E4F534A) throw std::runtime_error("no initial JSON header!");
        header = json::parse(file.begin() + 20, file.begin() + 20 + chunk_length);
        size_t pos = 20 + chunk_length, start = pos;
        std::deque<json> info;
        if (header.contains("buffers"))
            for (auto& b : header["buffers"]) info.push_back(b);
        while (pos - start < length) {
            if (!info.empty()) {
                json b = info.front();
                info.pop_front();
                if (b.contains("uri")) {
                    buffers.push_back(uri_to_bytes(b["uri"].get<std::string>(), dir));
                    continue;
                }
            }
            if (pos + 8 > file.size()) break;
            uint32_t cl, ctp;
            std::memcpy(&cl, &file[pos], 4);
            std::memcpy(&ctp, &file[pos + 4], 4);
            if (ctp != 0x004E4942) throw std::runtime_error("not binary GLTF!");
            pos += 8;
            if (pos + cl > file.size()) throw std::runtime_error("chunk was not expected length!");
            buffers.emplace_back(file.begin() + pos, file.begin() + pos + cl);
            pos += cl;
        }
    } else {
        header = json::parse(file.begin(), file.end());
        if (header.contains("buffers"))
            for (auto& b : header["buffers"]) {
                if (b.contains("uri")) buffers.push_back(uri_to_bytes(b["uri"].get<std::string>(), dir));
                else buffers.emplace_back();
            }
    }

    LoadedScene S;
    std::vector<std::vector<uint8_t>> views;
    std::vector<Accessor> access;
    std::vector<ImagePtr> images;
    std::vector<std::vector<uint8_t>> image_blobs;
    std::vector<char> decoded;
    if (header.contains("bufferViews")) {
        for (auto& v : header["bufferViews"]) {
            size_t s = v.value("byteOffset", (size_t)0);
            size_t len = v["byteLength"].get<size_t>();
            const auto& b = buffers.at(v["buffer"].get<int>());
            if (s + len > b.size()) throw std::runtime_error("bufferView out of range");
            views.emplace_back(b.begin() + s, b.begin() + s + len);
        }
        for (auto& a : header["accessors"]) {
            Accessor acc;
            acc.ct = a["componentType"].get<int>();
            acc.nc = ncomp(a["type"].get<std::string>());
            acc.count = a["count"].get<int64_t>();
            int is = itemsize(acc.ct);
            size_t row = (size_t)acc.nc * is;
            acc.bytes.assign(acc.count * row, 0);
            if (a.contains("bufferView")) {
                const json& bv = header["bufferViews"][a["bufferView"].get<int>()];
                const auto& data = views[a["bufferView"].get<int>()];
                size_t start = a.value("byteOffset", (size_t)0);
                if (bv.contains("byteStride")) {
                    size_t stride = bv["byteStride"].get<size_t>();
                    for (int64_t i = 0; i < acc.count; i++)
                        std::memcpy(&acc.bytes[i * row], &data.at(start + i * stride), row);
                } else {
                    if (start + acc.bytes.size() > data.size()) throw std::runtime_error("accessor out of range");
                    std::memcpy(acc.bytes.data(), &data[start], acc.bytes.size());
                }
            }
            access.push_back(std::move(acc));
        }
        // images (decoded lazily, like PIL.Image.open + convert on first use)
        if (header.contains("images")) {
            for (auto& img : header["images"]) {
                std::vector<uint8_t> blob;
                if (img.value("mimeType", std::string()) == "image/ktx2") {
                } else if (img.contains("bufferView")) {
                    blob = views[img["bufferView"].get<int>()];
                } else if (img.contains("uri")) {
                    try {
                        blob = uri_to_bytes(img["uri"].get<std::string>(), dir);
                    } catch (...) {
                    }
                }
                image_blobs.push_back(std::move(blob));
            }
            images.assign(image_blobs.size(), nullptr);
            decoded.assign(image_blobs.size(), 0);
        }
        if (header.contains("materials"))
            for (auto& m : header["materials"]) S.materials.push_back(parse_material(m, header, images, image_blobs, decoded));
    }

    // ---- meshes: one Trimesh per primitive (mode 4 / 5) ----------------------------------------
    std::vector<std::vector<int>> mesh_prim;
    if (header.contains("meshes")) {
        for (auto& m : header["meshes"]) {
            std::vector<int> names;
            for (auto& p : m["primitives"]) {
                if (p.contains("extensions") && p["extensions"].contains("KHR_draco_mesh_compression")) {
                    const json& ext = p["extensions"]["KHR_draco_mesh_compression"];
                    draco_decode(views.at(ext["bufferView"].get<int>()), ext, p, access);
                }
                int mode = p.value("mode", 4);
                if (mode != 4 && mode != 5) continue;  // lines / points are not Trimesh; fans unsupported
                const json& attr = p["attributes"];
                Geometry g;
                const Accessor& pa = access[attr["POSITION"].get<int>()];
                int64_t nv = pa.count;
                g.vertices.resize(nv * 3);
                for (int64_t i = 0; i < nv * 3; i++) g.vertices[i] = pa.get(i);
                std::vector<int64_t> flat;
                if (p.contains("indices")) {
                    const Accessor& ia = access[p["indices"].get<int>()];
                    for (int64_t i = 0; i < ia.size(); i++) flat.push_back(ia.geti(i));
                } else {
                    for (int64_t i = 0; i < nv; i++) flat.push_back(i);
                }
                if (mode == 5) {
                    for (int64_t i = 0; i + 2 < (int64_t)flat.size(); i++) {
                        if (i % 2 == 1) {
                            g.faces.push_back(flat[i + 2]);
                            g.faces.push_back(flat[i + 1]);
                            g.faces.push_back(flat[i]);
                        } else {
                            g.faces.push_back(flat[i]);
                            g.faces.push_back(flat[i + 1]);
                            g.faces.push_back(flat[i + 2]);
                        }
                    }
                } else {
                    g.faces.assign(flat.begin(), flat.begin() + (flat.size() / 3) * 3);
                }
                bool visuals = false;
                if (p.contains("material")) {
                    visuals = true;
                    g.visual.texture_visuals = true;
                    g.visual.material_key = p["material"].get<int>();
                    if (attr.contains("TEXCOORD_0")) {
                        const Accessor& ua = access[attr["TEXCOORD_0"].get<int>()];
                        g.visual.has_uv = true;
                        g.visual.uv.resize(ua.count * 2);
                        for (int64_t i = 0; i < ua.count; i++) {
                            g.visual.uv[i * 2] = ua.get(i * ua.nc);
                            if (ua.ct == 5126) {
                                float v = ua.raw<float>(i * ua.nc + 1);
                                g.visual.uv[i * 2 + 1] = (double)(1.0f - v);  // float32 arithmetic
                            } else {
                                g.visual.uv[i * 2 + 1] = 1.0 - ua.get(i * ua.nc + 1);
                            }
                        }
                    }
                }
                if (attr.contains("COLOR_0")) {
                    const Accessor& ca = access[attr["COLOR_0"].get<int>()];
                    if (ca.count == nv) {
                        Visual& vis = g.visual;
                        if (!visuals) {
                            vis.color_kind = 1;  // ColorVisuals(vertex_colors=to_rgba(colors))
                            vis.color_cols = 4;
                            if (ca.ct == 5126) {
                                std::vector<float> c(ca.size());
                                for (int64_t i = 0; i < ca.size(); i++) c[i] = ca.raw<float>(i);
                                vis.colors = to_rgba_f32(c, ca.nc);
                            } else {
                                vis.colors.assign(nv * 4, 255.0);
                                for (int64_t i = 0; i < nv; i++)
                                    for (int k = 0; k < ca.nc && k < 4; k++)
                                        vis.colors[i * 4 + k] = (double)(uint8_t)ca.geti(i * ca.nc + k);
                            }
                        } else {
                            vis.color_kind = 3;  // raw vertex_attributes['color']
                            vis.color_cols = ca.nc;
                            vis.colors.resize(ca.size());
                            for (int64_t i = 0; i < ca.size(); i++) vis.colors[i] = ca.get(i);
                        }
                    }
                }
                S.geometries.push_back(std::move(g));
                names.push_back((int)S.geometries.size() - 1);
            }
            mesh_prim.push_back(names);
        }
    }

    // ---- node graph (deque pop-right traversal) ----------------------------------------------
    const json nodes = header.value("nodes", json::array());
    // frames: -1 base, 0..N-1 nodes, N.. primitive child frames
    struct Edge {
        int64_t u, v;
        accel::Mat4 M;
        int geometry;
    };
    std::vector<Edge> graph;
    std::deque<std::pair<int64_t, int64_t>> queue;
    int scene_index = header.value("scene", 0);
    if (header.contains("scenes"))
        for (auto& r : header["scenes"][scene_index].value("nodes", json::array()))
            queue.push_back({-1, r.get<int64_t>()});
    std::set<std::pair<int64_t, int64_t>> consumed;
    bool have_camera = false;
    int64_t next_frame = (int64_t)nodes.size();
    while (!queue.empty()) {
        auto edge = queue.back();
        queue.pop_back();
        if (consumed.count(edge)) continue;
        consumed.insert(edge);
        int64_t a = edge.first, b = edge.second;
        const json& child = nodes[b];
        if (child.contains("children"))
            for (auto& c : child["children"]) queue.push_back({b, c.get<int64_t>()});
        accel::Mat4 M = accel::Mat4::eye();
        if (child.contains("matrix")) {
            std::vector<double> mv;
            for (auto& x : child["matrix"]) mv.push_back(x.get<double>());
            // np.array(m).reshape(4, 4).T: an F-ordered view
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++) M(r, c) = mv[c * 4 + r];
            M.forder = true;
        }
        if (child.contains("translation")) {
            accel::Mat4 T = accel::Mat4::eye();
            for (int k = 0; k < 3; k++) T(k, 3) = child["translation"][k].get<double>();
            M = accel::dot4(M, T);
        }
        if (child.contains("rotation")) {
            double q[4] = {child["rotation"][3].get<double>(), child["rotation"][0].get<double>(),
                           child["rotation"][1].get<double>(), child["rotation"][2].get<double>()};
            M = accel::dot4(M, accel::quaternion_matrix(q));
        }
        if (child.contains("scale")) {
            accel::Mat4 Sm = accel::Mat4::eye();
            for (int k = 0; k < 3; k++) Sm(k, k) = child["scale"][k].get<double>();
            M = accel::dot4(M, Sm);
        }
        if (child.contains("camera") && !have_camera) {
            try {
                const json& cam = header["cameras"][child["camera"].get<int>()];
                if (cam.contains("perspective") && cam["perspective"].contains("znear") &&
                    cam["perspective"].contains("aspectRatio") && cam["perspective"].contains("yfov"))
                    have_camera = true;
            } catch (...) {
            }
            continue;
        }
        if (child.contains("mesh")) {
            const auto& geoms = mesh_prim.at(child["mesh"].get<int>());
            if (geoms.size() > 1) {
                graph.push_back({a, b, M, -1});
                for (int gi : geoms) graph.push_back({b, next_frame++, accel::Mat4::eye(), gi});
            } else if (geoms.size() == 1) {
                graph.push_back({a, b, M, geoms[0]});
            }
        } else {
            graph.push_back({a, b, M, -1});
        }
    }

    // ---- scene graph (EnforcedForest) ---------------------------------------------------------
    std::map<std::pair<int64_t, int64_t>, size_t> edge_data;  // -> index in graph
    std::map<int64_t, int64_t> parents;
    std::vector<int64_t> node_order;
    std::map<int64_t, int> node_geom;
    std::set<int64_t> seen;
    auto touch = [&](int64_t n) {
        if (!seen.count(n)) {
            seen.insert(n);
            node_order.push_back(n);
        }
    };
    touch(-1);
    for (size_t i = 0; i < graph.size(); i++) {
        const Edge& e = graph[i];
        auto key = std::make_pair(e.u, e.v);
        auto it = edge_data.find(key);
        if (it != edge_data.end()) {
            const Edge& old = graph[it->second];
            bool close = true;
            for (int k = 0; k < 16; k++) {
                double x = e.M.m[k], y = old.M.m[k];
                if (!(std::fabs(x - y) <= 1e-8 + 1e-8 * std::fabs(y))) close = false;
            }
            if (close && old.geometry == e.geometry) continue;
        }
        parents[e.v] = e.u;
        edge_data[key] = i;
        touch(e.u);
        touch(e.v);
        if (e.geometry >= 0) node_geom[e.v] = e.geometry;
    }
    for (int64_t n : node_order) {
        auto g = node_geom.find(n);
        if (g == node_geom.end()) continue;
        accel::Mat4 M;
        auto direct = edge_data.find({-1, n});
        if (direct != edge_data.end()) {
            M = graph[direct->second].M;
        } else {
            std::vector<int64_t> path{n};
            while (path.back() != -1) {
                auto p = parents.find(path.back());
                if (p == parents.end()) throw std::runtime_error("disconnected scene graph node");
                path.push_back(p->second);
            }
            std::reverse(path.begin(), path.end());
            std::vector<accel::Mat4> mats;
            for (size_t i = 0; i + 1 < path.size(); i++) {
                const accel::Mat4& m = graph[edge_data.at({path[i], path[i + 1]})].M;
                double dev = 0;
                for (int k = 0; k < 16; k++) dev = np::fmax0(dev, std::fabs(m.m[k] - ((k % 5 == 0) ? 1.0 : 0.0)));
                if (dev > 1e-8) mats.push_back(m);
            }
            if (mats.empty()) M = accel::Mat4::eye();
            else if (mats.size() == 1) M = mats[0];
            else {
                M = mats.back();
                for (int i = (int)mats.size() - 2; i >= 0; i--) M = accel::dot4(mats[i], M);
            }
        }
        M = accel::fix_rigid(M, 1e-5);
        SceneNode sn;
        std::memcpy(sn.T, M.m, sizeof(sn.T));
        sn.forder = M.forder;
        sn.geometry = g->second;
        S.nodes.push_back(sn);
    }
    return S;
}

}  // namespace faqem
