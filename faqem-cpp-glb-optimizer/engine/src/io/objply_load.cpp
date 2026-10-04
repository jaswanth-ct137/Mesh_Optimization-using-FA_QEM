// OBJ and PLY loading with trimesh 5.1.0 semantics (process=False), including its vertex
// splitting by uv / normal (visual.texture.unmerge_faces + grouping.unique_rows).
#include <cmath>
#include <cstring>
#include <map>
#include <regex>
#include <stdexcept>

#include "io/asset.hpp"

namespace faqem {

VecI triangulate_quads(const std::vector<VecI>& polys);

namespace {

std::string strip(const std::string& s, const char* ws = " \t\n\r\f\v") {
    size_t a = s.find_first_not_of(ws);
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(ws);
    return s.substr(a, b - a + 1);
}
std::vector<std::string> split_ws(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && isspace((unsigned char)s[i])) i++;
        size_t j = i;
        while (j < s.size() && !isspace((unsigned char)s[j])) j++;
        if (j > i) out.push_back(s.substr(i, j - i));
        i = j;
    }
    return out;
}
std::string replace_all(std::string s, const std::string& a, const std::string& b) {
    size_t p = 0;
    while ((p = s.find(a, p)) != std::string::npos) {
        s.replace(p, a.size(), b);
        p += b.size();
    }
    return s;
}
std::vector<std::string> split_str(const std::string& s, const std::string& sep) {
    std::vector<std::string> out;
    size_t p = 0;
    while (true) {
        size_t q = s.find(sep, p);
        if (q == std::string::npos) {
            out.push_back(s.substr(p));
            break;
        }
        out.push_back(s.substr(p, q - p));
        p = q + sep.size();
    }
    return out;
}
std::vector<std::string> splitlines(const std::string& s) {
    std::vector<std::string> out;
    size_t p = 0;
    while (p < s.size()) {
        size_t q = s.find_first_of("\n\r\v\f", p);
        if (q == std::string::npos) {
            out.push_back(s.substr(p));
            break;
        }
        out.push_back(s.substr(p, q - p));
        p = q + 1;
        if (s[q] == '\r' && p < s.size() && s[p] == '\n') p++;
    }
    return out;
}
std::string first_line(const std::string& s) {
    size_t n = s.find('\n');
    return n == std::string::npos ? s : s.substr(0, n);
}
// np.fromstring(text, sep=" ") as float64 / int64
VecD fromstring_f(const std::string& t) {
    VecD out;
    for (const auto& tok : split_ws(t)) {
        char* e = nullptr;
        double v = std::strtod(tok.c_str(), &e);
        if (e == tok.c_str() || *e) break;
        out.push_back(v);
    }
    return out;
}
VecI fromstring_i(const std::string& t) {
    VecI out;
    for (const auto& tok : split_ws(t)) {
        char* e = nullptr;
        long long v = std::strtoll(tok.c_str(), &e, 10);
        if (e == tok.c_str() || *e) break;
        out.push_back(v);
    }
    return out;
}
inline int64_t wrap(int64_t i, int64_t n) { return i < 0 ? i + n : i; }

// grouping.hashable_rows for an (n, k) int64 array (k <= 4: bit-packed uint64, else raw bytes)
struct HashRows {
    bool packed = false;
    std::vector<uint64_t> h;
    std::vector<std::string> bytes;
};
HashRows hashable_rows(const VecI& a, int k) {
    HashRows r;
    int64_t n = (int64_t)a.size() / k;
    if (n == 0) {
        r.packed = true;
        return r;
    }
    if (k == 1) {
        r.packed = true;
        for (int64_t i = 0; i < n; i++) r.h.push_back((uint64_t)a[i]);
        return r;
    }
    if (k <= 4) {
        int precision = 64 / k;
        int64_t dmin = a[0], dmax = a[0];
        for (int64_t v : a) {
            dmin = std::min(dmin, v);
            dmax = std::max(dmax, v);
        }
        int64_t threshold = ((int64_t)1 << (precision - 1)) - 1;
        if (dmax < threshold && dmin > -threshold) {
            r.packed = true;
            r.h.assign(n, 0);
            for (int c = 0; c < k; c++)
                for (int64_t i = 0; i < n; i++)
                    r.h[i] ^= ((uint64_t)(a[i * k + c] + threshold + 1)) << (c * precision);
            return r;
        }
    }
    r.packed = false;
    for (int64_t i = 0; i < n; i++) r.bytes.emplace_back((const char*)&a[i * k], (size_t)k * 8);
    return r;
}
// np.unique(rows, return_index=True, return_inverse=True)
void unique_hash(const HashRows& r, VecI& index, VecI& inverse) {
    int64_t n = r.packed ? (int64_t)r.h.size() : (int64_t)r.bytes.size();
    VecI order(n);
    for (int64_t i = 0; i < n; i++) order[i] = i;
    if (r.packed)
        std::stable_sort(order.begin(), order.end(), [&](int64_t x, int64_t y) { return r.h[x] < r.h[y]; });
    else
        std::stable_sort(order.begin(), order.end(), [&](int64_t x, int64_t y) {
            return std::memcmp(r.bytes[x].data(), r.bytes[y].data(), r.bytes[x].size()) < 0;
        });
    index.clear();
    inverse.assign(n, 0);
    int64_t g = -1;
    for (int64_t i = 0; i < n; i++) {
        bool fresh = i == 0 || (r.packed ? r.h[order[i]] != r.h[order[i - 1]] : r.bytes[order[i]] != r.bytes[order[i - 1]]);
        if (fresh) {
            g++;
            index.push_back(order[i]);
        }
        inverse[order[i]] = g;
    }
}
// visual.texture.unmerge_faces(faces, *args): returns new faces and one mask per input
VecI unmerge_faces(const VecI& faces, int width, const std::vector<const VecI*>& args, std::vector<VecI>& masks) {
    int k = 1 + (int)args.size();
    int64_t n = (int64_t)faces.size();
    VecI stack(n * k);
    for (int64_t i = 0; i < n; i++) {
        stack[i * k] = faces[i];
        for (int a = 0; a < (int)args.size(); a++) stack[i * k + 1 + a] = (*args[a])[i];
    }
    VecI unique, inverse;
    unique_hash(hashable_rows(stack, k), unique, inverse);
    int64_t u = (int64_t)unique.size();
    VecI first(u);
    for (int64_t i = 0; i < u; i++) first[i] = stack[unique[i] * k];
    VecI order = np::argsort_quick(first);
    VecI remap(u);
    for (int64_t i = 0; i < u; i++) remap[order[i]] = i;
    VecI nf(n);
    for (int64_t i = 0; i < n; i++) nf[i] = remap[inverse[i]];
    masks.assign(k, VecI(u));
    for (int64_t i = 0; i < u; i++)
        for (int c = 0; c < k; c++) masks[c][i] = stack[unique[order[i]] * k + c];
    (void)width;
    return nf;
}

// Trimesh faces setter: (n, 4) quads -> triangles
VecI faces_setter(const VecI& f, int width) {
    if (width == 3) return f;
    std::vector<VecI> polys;
    for (size_t i = 0; i < f.size(); i += width) polys.emplace_back(f.begin() + i, f.begin() + i + width);
    return triangulate_quads(polys);
}

std::string dirname(const std::string& p) {
    size_t s = p.find_last_of("/\\");
    return s == std::string::npos ? std::string(".") : p.substr(0, s);
}

uint8_t to_u8(double c) { return (uint8_t)np::rint(np::clip(c * 255, 0.0, 255.0)); }

}  // namespace

// =================================================================================================
// OBJ
// =================================================================================================
LoadedScene load_obj_scene(const std::string& path) {
    std::vector<uint8_t> raw = read_file(path);
    std::string text = std::string(raw.begin(), raw.end());
    text = "\n" + replace_all(strip(text), "\r\n", "\n") + "\n";
    text = replace_all(text, "\\\n", "");
    LoadedScene S;
    std::string dir = dirname(path);

    // ---- materials (parse_mtl -> SimpleMaterial; any failure drops them all) ----------------
    std::map<std::string, int> materials;  // name -> material key
    size_t mp = text.find("mtllib");
    if (mp != std::string::npos) {
        size_t nl = text.find('\n', mp);
        std::string mtl_path = strip(text.substr(mp + 6, nl - (mp + 6)));
        try {
            std::vector<uint8_t> mraw = read_file(dir + "/" + mtl_path);
            std::string mtl = strip(std::string(mraw.begin(), mraw.end()));
            struct M {
                std::string name;
                std::vector<double> kd;
                bool has_kd = false, kd_scalar = false, bad_color = false;
                double ns = 1.0;
                bool has_ns = false, ns_list = false;
                ImagePtr image;
            };
            std::vector<M> mats;
            std::map<std::string, size_t> by_name;
            M* cur = nullptr;
            M tmp;
            bool have = false;
            auto commit = [&]() {
                if (!have) return;
                if (by_name.count(tmp.name)) mats[by_name[tmp.name]] = tmp;
                else {
                    by_name[tmp.name] = mats.size();
                    mats.push_back(tmp);
                }
            };
            for (const std::string& line : splitlines(mtl)) {
                auto sp = split_ws(strip(line));
                if (sp.size() <= 1) continue;
                std::string key = sp[0];
                for (auto& c : key) c = (char)tolower(c);
                if (key == "newmtl") {
                    commit();
                    tmp = M();
                    have = true;
                    std::string nm;
                    for (size_t i = 1; i < sp.size(); i++) nm += (i > 1 ? " " : "") + sp[i];
                    tmp.name = nm;
                    cur = &tmp;
                } else if (key == "map_kd") {
                    std::string low = line;
                    for (auto& c : low) c = (char)tolower(c);
                    std::string fn = strip(line.substr(low.find("map_kd") + 6));
                    try {
                        std::vector<uint8_t> img = read_file(dir + "/" + fn);
                        if (cur) cur->image = decode_image_rgba(img.data(), img.size());
                    } catch (...) {
                    }
                } else if (key == "ka" || key == "ks") {
                    // parsed like Kd; SimpleMaterial's to_rgba raises on anything but 3 / 4 values
                    size_t nvals = 0;
                    bool okp = true;
                    for (size_t i = 1; i < sp.size(); i++) {
                        char* e = nullptr;
                        std::strtod(sp[i].c_str(), &e);
                        if (e == sp[i].c_str() || *e) {
                            okp = false;
                            break;
                        }
                        nvals++;
                    }
                    if (okp && cur && nvals != 3 && nvals != 4) cur->bad_color = true;
                } else if (key == "kd" || key == "ns") {
                    std::vector<double> vals;
                    bool okp = true;
                    for (size_t i = 1; i < sp.size(); i++) {
                        char* e = nullptr;
                        double v = std::strtod(sp[i].c_str(), &e);
                        if (e == sp[i].c_str() || *e) {
                            okp = false;
                            break;
                        }
                        vals.push_back(v);
                    }
                    if (okp && cur) {
                        if (key == "kd") {
                            cur->kd = vals;
                            cur->has_kd = true;
                            cur->kd_scalar = vals.size() == 1;
                        } else {
                            cur->has_ns = true;
                            cur->ns_list = vals.size() != 1;
                            cur->ns = vals[0];
                        }
                    }
                }
            }
            commit();
            // SimpleMaterial(**kwargs) for every material; to_rgba / float() failures drop all
            std::vector<Material> built;
            for (auto& m : mats) {
                if (m.has_kd && (m.kd_scalar || (m.kd.size() != 3 && m.kd.size() != 4)))
                    throw std::runtime_error("bad Kd");
                if (m.has_ns && m.ns_list) throw std::runtime_error("bad Ns");
                if (m.bad_color) throw std::runtime_error("bad Ka / Ks");
                Material pm;
                // diffuse = to_rgba(Kd) (or DEFAULT_COLOR), roughness = (2 / (Ns + 2)) ** 0.25
                double rgba[4] = {102, 102, 102, 255};
                if (m.has_kd) {
                    for (size_t k = 0; k < m.kd.size(); k++) rgba[k] = to_u8(m.kd[k]);
                    if (m.kd.size() == 3) rgba[3] = 255;
                }
                for (int k = 0; k < 4; k++) pm.color[k] = rgba[k] / 255.0;
                double gl = m.has_ns ? m.ns : 1.0;
                pm.roughness = np::py_pow(2 / (gl + 2), 1.0 / 4.0);
                pm.metallic = 0.0;
                pm.texture = m.image;
                built.push_back(pm);
            }
            for (size_t i = 0; i < mats.size(); i++) {
                materials[mats[i].name] = (int)S.materials.size();
                S.materials.push_back(built[i]);
            }
        } catch (...) {
            materials.clear();
            S.materials.clear();
        }
    }

    // ---- vertices (_parse_vertices) ---------------------------------------------------------------
    const char* kinds[3] = {"v", "vt", "vn"};
    std::map<std::string, std::pair<VecD, int>> result;  // key -> (data, columns)
    {
        std::map<std::string, size_t> starts, ends;
        bool any = false;
        for (auto k : kinds) {
            size_t s = text.find(std::string("\n") + k + " ");
            if (s != std::string::npos) {
                starts[k] = s;
                any = true;
            }
        }
        if (any) {
            size_t start = std::string::npos, end = 0;
            bool have_end = false;
            for (auto& kv : starts) {
                start = std::min(start, kv.second);
                size_t r = text.rfind("\n" + kv.first + " ");
                size_t e = text.find('\n', r + 2 + kv.first.size());
                if (e != std::string::npos) {
                    end = have_end ? std::max(end, e) : e;
                    have_end = true;
                }
            }
            std::string chunk = text.substr(start, end - start);
            chunk = replace_all(replace_all(chunk, "+e", "e"), "-e", "e");
            for (auto& kv : starts) {
                auto parts = split_str(chunk, "\n" + kv.first + " ");
                std::vector<std::string> value;
                for (size_t i = 1; i < parts.size(); i++) value.push_back(first_line(parts[i]));
                if (value.empty()) continue;
                int per_row = (int)split_ws(value[0]).size();
                std::string joined;
                for (size_t i = 0; i < value.size(); i++) joined += (i ? " " : "") + value[i];
                VecD arr = fromstring_f(joined);
                if ((int64_t)arr.size() == (int64_t)value.size() * per_row) {
                    result[kv.first] = {arr, per_row};
                } else {
                    std::vector<std::vector<std::string>> lines;
                    for (auto& v : value)
                        for (auto& l : splitlines(v)) lines.push_back(split_ws(strip(l)));
                    size_t count = SIZE_MAX;
                    for (auto& l : lines) count = std::min(count, l.size());
                    VecD a;
                    for (auto& l : lines)
                        for (size_t c = 0; c < count; c++) a.push_back(std::strtod(l[c].c_str(), nullptr));
                    result[kv.first] = {a, (int)count};
                }
            }
        }
    }
    VecD v, vc, vt, vn;
    int vcols = 0;
    bool have_v = result.count("v"), have_vt = result.count("vt"), have_vn = result.count("vn");
    if (have_v) {
        auto& r = result["v"];
        int64_t n = r.second ? (int64_t)r.first.size() / r.second : 0;
        vcols = r.second;
        for (int64_t i = 0; i < n; i++) {
            for (int k = 0; k < 3; k++) v.push_back(r.first[i * vcols + k]);
            if (vcols >= 6)
                for (int k = 3; k < 6; k++) vc.push_back(r.first[i * vcols + k]);
        }
    }
    if (have_vt) {
        auto& r = result["vt"];
        int64_t n = (int64_t)r.first.size() / r.second;
        for (int64_t i = 0; i < n; i++)
            for (int k = 0; k < 2; k++) vt.push_back(r.first[i * r.second + k]);
    }
    if (have_vn) vn = result["vn"].first;
    int64_t nvtx = (int64_t)v.size() / 3, nvt = (int64_t)vt.size() / 2;

    // ---- faces (_preprocess_faces + _group_by) ---------------------------------------------------
    const char* starters[5] = {"\nusemtl ", "\no ", "\nf ", "\ng ", "\ns "};
    size_t f_start = text.size();
    for (auto st : starters) {
        size_t s = text.find(st);
        if (s != std::string::npos && s < f_start) f_start = s;
    }
    size_t rf = text.rfind("\nf ");
    size_t f_end = rf == std::string::npos ? text.find('\n', 2) : text.find('\n', rf + 3);
    std::string f_chunk = f_end != std::string::npos ? text.substr(f_start, f_end - f_start) : text.substr(f_start);
    VecI splits{0, (int64_t)f_chunk.size()};
    for (size_t p = f_chunk.find("usemtl "); p != std::string::npos; p = f_chunk.find("usemtl ", p + 1))
        splits.push_back((int64_t)p);
    std::sort(splits.begin(), splits.end());
    splits.erase(std::unique(splits.begin(), splits.end()), splits.end());
    struct Tuple {
        bool has_mtl;
        std::string mtl;
        std::string chunk;
    };
    std::vector<Tuple> tuples;
    bool has_mtl = false;
    std::string cur_mtl;
    for (size_t i = 0; i + 1 < splits.size(); i++) {
        std::string chunk = strip(f_chunk.substr(splits[i], splits[i + 1] - splits[i])) + "\n";
        if (chunk.rfind("usemtl", 0) == 0) {
            size_t nl = chunk.find('\n');
            cur_mtl = strip(chunk.substr(6, nl - 6));
            has_mtl = true;
            chunk = chunk.substr(nl + 1);
        }
        if (chunk.rfind("f ", 0) == 0 || chunk.find("\nf") != std::string::npos) tuples.push_back({has_mtl, cur_mtl, chunk});
    }
    // _group_by(use_mtl=True): group chunks by material in first-appearance order
    std::vector<Tuple> grouped;
    for (auto& t : tuples) {
        bool found = false;
        for (auto& g : grouped)
            if (g.has_mtl == t.has_mtl && g.mtl == t.mtl) {
                g.chunk += "\n" + t.chunk;
                found = true;
                break;
            }
        if (!found) grouped.push_back(t);
    }
    if (grouped.empty()) throw std::runtime_error("No triangle mesh found in the file.");

    while (!grouped.empty()) {
        Tuple t = grouped.back();
        grouped.pop_back();
        // face lines: re.split("^f", chunk, flags=MULTILINE)[1:]
        std::vector<std::string> face_lines;
        {
            std::vector<size_t> pos;
            if (!t.chunk.empty() && t.chunk[0] == 'f') pos.push_back(0);
            for (size_t p = t.chunk.find("\nf"); p != std::string::npos; p = t.chunk.find("\nf", p + 1)) pos.push_back(p + 1);
            for (size_t i = 0; i < pos.size(); i++) {
                size_t s = pos[i] + 1;
                size_t e = i + 1 < pos.size() ? pos[i + 1] : t.chunk.size();
                face_lines.push_back(strip(first_line(t.chunk.substr(s, e - s))));
            }
        }
        auto ncols = [](const std::string& f) { return (int)split_ws(replace_all(f, "/", " ")).size(); };
        int columns = ncols(face_lines[0]);
        bool flat = true;
        for (auto& f : face_lines) flat = flat && ncols(f) == columns;
        VecI faces, ftex, fnorm;
        bool has_tex = false, has_norm = false;
        int width = 3;
        if (flat) {
            std::string joined;
            for (size_t i = 0; i < face_lines.size(); i++) joined += (i ? " " : "") + face_lines[i];
            VecI arr = fromstring_i(replace_all(joined, "/", " "));
            for (auto& x : arr)
                if (x > 0) x -= 1;
            int group_count = (int)split_ws(strip(face_lines[0])).size();
            int per_ref = columns / group_count;
            int64_t rows = (int64_t)arr.size() / columns;
            width = group_count;
            auto col = [&](int off) {
                VecI o;
                for (int64_t r = 0; r < rows; r++)
                    for (int g = 0; g < group_count; g++) o.push_back(arr[r * columns + g * per_ref + off]);
                return o;
            };
            faces = col(0);
            if (columns == group_count * 2) {
                std::string s;
                for (auto& tok : split_ws(face_lines[0])) s += strip(tok, "/");
                int64_t cnt = std::count(s.begin(), s.end(), '/');
                if (cnt == columns) {
                    fnorm = col(1);
                    has_norm = true;
                } else if (cnt == columns / 2) {
                    ftex = col(1);
                    has_tex = true;
                }
            } else if (columns == group_count * 3) {
                ftex = col(1);
                fnorm = col(2);
                has_tex = has_norm = true;
            }
        } else {
            VecI fv, fvt, fvn;
            for (auto& line : face_lines) {
                auto sp = split_ws(first_line(strip(line)));
                std::vector<std::string> s2;
                if (sp.size() == 3) s2 = sp;
                else if (sp.size() == 4) s2 = {sp[0], sp[1], sp[2], sp[2], sp[3], sp[0]};
                else if (sp.size() > 4) {
                    for (size_t i = 0; i + 2 < sp.size(); i++) s2.insert(s2.end(), {sp[0], sp[i + 1], sp[i + 2]});
                } else continue;
                for (auto& f : s2) {
                    auto parts = split_str(f, "/");
                    fv.push_back(std::stoll(parts[0]));
                    try {
                        if (parts.size() > 1) fvt.push_back(std::stoll(parts[1]));
                    } catch (...) {
                    }
                    try {
                        if (parts.size() > 2) fvn.push_back(std::stoll(parts[2]));
                    } catch (...) {
                    }
                }
            }
            auto dec = [](VecI& a) {
                for (auto& x : a)
                    if (x > 0) x -= 1;
            };
            faces = fv;
            dec(faces);
            if (fvt.size() == fv.size()) {
                ftex = fvt;
                dec(ftex);
                has_tex = true;
            }
            if (fvn.size() == fv.size()) {
                fnorm = fvn;
                dec(fnorm);
                has_norm = true;
            }
        }
        Geometry g;
        VecI mask_v;
        VecI new_faces;
        bool has_uv = false;
        VecD uv;
        if (has_tex) {
            std::vector<VecI> masks;
            if (has_norm && fnorm.size() == faces.size()) new_faces = unmerge_faces(faces, width, {&ftex, &fnorm}, masks);
            else new_faces = unmerge_faces(faces, width, {&ftex}, masks);
            mask_v = masks[0];
            bool okuv = true;
            for (int64_t i : masks[1])
                if (wrap(i, nvt) < 0 || wrap(i, nvt) >= nvt) okuv = false;
            if (okuv) {
                has_uv = true;
                for (int64_t i : masks[1]) {
                    uv.push_back(vt[wrap(i, nvt) * 2]);
                    uv.push_back(vt[wrap(i, nvt) * 2 + 1]);
                }
            }
        } else if (have_vn && has_norm && fnorm.size() == faces.size()) {
            std::vector<VecI> masks;
            new_faces = unmerge_faces(faces, width, {&fnorm}, masks);
            mask_v = masks[0];
        } else {
            std::vector<char> used(nvtx, 0);
            for (int64_t f : faces) used[wrap(f, nvtx)] = 1;
            VecI inv(nvtx, 0);
            int64_t c = 0;
            for (int64_t i = 0; i < nvtx; i++)
                if (used[i]) {
                    inv[i] = c++;
                    mask_v.push_back(i);
                }
            for (int64_t f : faces) new_faces.push_back(inv[wrap(f, nvtx)]);
        }
        for (int64_t i : mask_v)
            for (int k = 0; k < 3; k++) g.vertices.push_back(v[wrap(i, nvtx) * 3 + k]);
        g.faces = faces_setter(new_faces, width);
        int64_t gv = (int64_t)g.vertices.size() / 3;
        if (t.has_mtl && materials.count(t.mtl)) {
            g.visual.texture_visuals = true;
            g.visual.material_key = materials[t.mtl];
            g.visual.has_uv = has_uv;
            g.visual.uv = uv;
        } else if (has_uv && (int64_t)uv.size() / 2 == gv) {
            // TextureVisuals(uv=uv): a fresh empty_material (2x2 grey image, DEFAULT_COLOR diffuse)
            Material em;
            ImagePtr img = std::make_shared<Image>();
            img->w = img->h = 2;
            img->channels = 4;
            img->px = {100, 100, 100, 255, 100, 100, 100, 255, 100, 100, 100, 255, 100, 100, 100, 255};
            em.texture = img;
            for (int k = 0; k < 3; k++) em.color[k] = 102 / 255.0;
            em.color[3] = 1.0;
            em.roughness = np::py_pow(2 / (1.0 + 2), 1.0 / 4.0);
            g.visual.texture_visuals = true;
            g.visual.material_key = (int)S.materials.size();
            S.materials.push_back(em);
            g.visual.has_uv = true;
            g.visual.uv = uv;
        } else if (!vc.empty()) {
            // ColorVisuals(vertex_colors=to_rgba(vc[mask_v])) (float64)
            g.visual.color_kind = 1;
            g.visual.color_cols = 4;
            for (int64_t i : mask_v) {
                for (int k = 0; k < 3; k++) g.visual.colors.push_back((double)to_u8(vc[wrap(i, nvtx) * 3 + k]));
                g.visual.colors.push_back(255.0);
            }
        }
        S.geometries.push_back(std::move(g));
        SceneNode n;
        for (int i = 0; i < 16; i++) n.T[i] = (i % 5 == 0) ? 1.0 : 0.0;
        n.forder = false;
        n.geometry = (int)S.geometries.size() - 1;
        S.nodes.push_back(n);
    }
    return S;
}

// =================================================================================================
// PLY
// =================================================================================================
namespace {
struct Prop {
    std::string name, type, count_type;
    bool list = false;
    int64_t list_len = -1;  // binary: fixed from the first row
};
struct Element {
    std::string name;
    int64_t length = 0;
    std::vector<Prop> props;
    // parsed data: per property, rows x cols as float64 (values after astype(dtype))
    std::map<std::string, std::pair<VecD, int64_t>> data;  // name -> (values, cols)
    std::map<std::string, std::vector<VecD>> ragged;
};
int type_size(const std::string& t) {
    static const std::map<std::string, int> m = {
        {"char", 1}, {"uchar", 1}, {"short", 2}, {"ushort", 2}, {"int", 4}, {"int8", 1}, {"int16", 2},
        {"int32", 4}, {"int64", 8}, {"uint", 4}, {"uint8", 1}, {"uint16", 2}, {"uint32", 4}, {"uint64", 8},
        {"float", 4}, {"float16", 2}, {"float32", 4}, {"float64", 8}, {"double", 8}};
    auto it = m.find(t);
    if (it == m.end()) throw std::runtime_error("unknown PLY type " + t);
    return it->second;
}
// value of type t from raw bytes (endianness handled by the caller)
double read_typed(const uint8_t* p, const std::string& t, bool big) {
    uint8_t b[8];
    int n = type_size(t);
    for (int i = 0; i < n; i++) b[i] = big ? p[n - 1 - i] : p[i];
    if (t == "char" || t == "int8") return (int8_t)b[0];
    if (t == "uchar" || t == "uint8") return b[0];
    if (t == "short" || t == "int16") { int16_t v; std::memcpy(&v, b, 2); return v; }
    if (t == "ushort" || t == "uint16") { uint16_t v; std::memcpy(&v, b, 2); return v; }
    if (t == "int" || t == "int32") { int32_t v; std::memcpy(&v, b, 4); return v; }
    if (t == "uint" || t == "uint32") { uint32_t v; std::memcpy(&v, b, 4); return v; }
    if (t == "int64") { int64_t v; std::memcpy(&v, b, 8); return (double)v; }
    if (t == "uint64") { uint64_t v; std::memcpy(&v, b, 8); return (double)v; }
    if (t == "float" || t == "float32") { float v; std::memcpy(&v, b, 4); return v; }
    if (t == "double" || t == "float64") { double v; std::memcpy(&v, b, 8); return v; }
    throw std::runtime_error("unsupported PLY type " + t);
}
// float64 -> astype(type) -> float64
double cast_typed(double x, const std::string& t) {
    if (t == "float" || t == "float32") return (double)(float)x;
    if (t == "double" || t == "float64") return x;
    if (t == "char" || t == "int8") return (double)(int8_t)(int64_t)x;
    if (t == "uchar" || t == "uint8") return (double)(uint8_t)(int64_t)x;
    if (t == "short" || t == "int16") return (double)(int16_t)(int64_t)x;
    if (t == "ushort" || t == "uint16") return (double)(uint16_t)(int64_t)x;
    if (t == "int" || t == "int32") return (double)(int32_t)(int64_t)x;
    if (t == "uint" || t == "uint32") return (double)(uint32_t)(int64_t)x;
    return (double)(int64_t)x;
}
bool is_int_type(const std::string& t) { return t.find("float") == std::string::npos && t != "double"; }
}  // namespace

LoadedScene load_ply_scene(const std::string& path) {
    std::vector<uint8_t> d = read_file(path);
    size_t pos = 0;
    auto readline = [&]() {
        size_t e = pos;
        while (e < d.size() && d[e] != '\n') e++;
        std::string l(d.begin() + pos, d.begin() + e);
        pos = std::min(d.size(), e + 1);
        return l;
    };
    std::string l0 = readline();
    std::string ll = l0;
    for (auto& c : ll) c = (char)tolower(c);
    if (ll.find("ply") == std::string::npos) throw std::runtime_error("Not a ply file!");
    std::string enc = strip(readline());
    for (auto& c : enc) c = (char)tolower(c);
    bool ascii = enc.find("ascii") != std::string::npos;
    bool big = enc.find("big") != std::string::npos;
    std::vector<Element> els;
    while (true) {
        if (pos >= d.size()) throw std::runtime_error("Header not terminated properly!");
        std::string raw = strip(readline());
        auto line = split_ws(raw);
        if (std::find(line.begin(), line.end(), "end_header") != line.end()) break;
        if (line.empty()) continue;
        if (line[0].find("element") != std::string::npos) {
            Element e;
            e.name = line.at(1);
            e.length = std::stoll(line.at(2));
            els.push_back(e);
        } else if (line[0].find("property") != std::string::npos) {
            if (els.empty()) throw std::runtime_error("Property defined before any element!");
            Prop p;
            if (line.size() == 3) {
                p.type = line[1];
                p.name = line[2];
                type_size(p.type);
            } else if (line.size() > 1 && line[1].find("list") != std::string::npos) {
                p.list = true;
                p.count_type = line.at(2);
                p.type = line.at(3);
                p.name = line.at(4);
            } else {
                continue;
            }
            // OrderedDict: a repeated name replaces the earlier entry in place
            bool found = false;
            for (auto& q : els.back().props)
                if (q.name == p.name) {
                    q = p;
                    found = true;
                }
            if (!found) els.back().props.push_back(p);
        } else {
            std::string lr = raw;
            for (auto& c : lr) c = (char)tolower(c);
            if (lr.find("texturefile") != std::string::npos)
                throw std::runtime_error("unsupported: PLY with a texture file");
        }
    }
    if (ascii) {
        std::string text(d.begin() + pos, d.end());
        std::vector<VecD> rows;
        for (auto& l : splitlines(text)) rows.push_back(fromstring_f(l));
        size_t row = 0;
        for (auto& e : els) {
            if (e.length == 0) continue;
            std::vector<VecD> data(rows.begin() + std::min(rows.size(), row),
                                   rows.begin() + std::min(rows.size(), row + (size_t)e.length));
            row += e.length;
            bool equal = !data.empty();
            for (auto& r : data) equal = equal && r.size() == data[0].size();
            int nlist = 0;
            for (auto& p : e.props) nlist += p.list;
            if (equal && nlist <= 1) {
                // _load_element_single
                const VecD& first = data[0];
                size_t cur = 0;
                for (auto& p : e.props) {
                    if (cur >= first.size()) break;
                    int64_t len = 1;
                    size_t s = cur;
                    if (p.list) {
                        len = (int64_t)first[cur];
                        s = cur + 1;
                        cur += len + 1;
                    } else {
                        cur += 1;
                    }
                    VecD vals;
                    int64_t cols = 0;
                    for (auto& r : data) {
                        int64_t c = 0;
                        for (int64_t k = 0; k < len && s + k < r.size(); k++, c++) vals.push_back(cast_typed(r[s + k], p.type));
                        cols = c;
                    }
                    e.data[p.name] = {vals, cols};
                }
            } else {
                // _load_element_different: ragged rows (faces -> triangulate_quads)
                for (auto& r : data) {
                    size_t start = 0;
                    for (auto& p : e.props) {
                        int64_t len = 1;
                        if (p.list) {
                            len = (int64_t)r.at(start);
                            start += 1;
                        }
                        VecD vals;
                        for (int64_t k = 0; k < len && start + k < r.size(); k++) vals.push_back(cast_typed(r[start + k], p.type));
                        e.ragged[p.name].push_back(vals);
                        start += len;
                    }
                }
                for (auto& kv : e.ragged) {
                    bool same = true;
                    for (auto& r : kv.second) same = same && r.size() == kv.second[0].size();
                    if (same && !kv.second.empty() && kv.second[0].size() >= 1) {
                        VecD vals;
                        for (auto& r : kv.second) vals.insert(vals.end(), r.begin(), r.end());
                        e.data[kv.first] = {vals, (int64_t)kv.second[0].size()};
                    }
                }
            }
        }
    } else {
        // populate_listsize: list lengths come from the first row of each element
        size_t p_current = pos;
        for (auto& e : els) {
            size_t prior = 0;
            for (auto& p : e.props) {
                if (p.list) {
                    if (p_current + prior + type_size(p.count_type) > d.size()) throw std::runtime_error("PLY is unexpected length!");
                    p.list_len = (int64_t)read_typed(&d[p_current + prior], p.count_type, big);
                    prior += type_size(p.count_type) + p.list_len * type_size(p.type);
                } else {
                    prior += type_size(p.type);
                }
            }
            p_current += e.length * prior;
        }
        if (p_current != d.size()) throw std::runtime_error("PLY is unexpected length!");
        size_t off = pos;
        for (auto& e : els) {
            size_t rowsize = 0;
            for (auto& p : e.props) rowsize += p.list ? type_size(p.count_type) + p.list_len * type_size(p.type) : type_size(p.type);
            for (auto& p : e.props) e.data[p.name] = {VecD(), p.list ? p.list_len : 1};
            for (int64_t r = 0; r < e.length; r++) {
                size_t o = off + r * rowsize;
                for (auto& p : e.props) {
                    if (p.list) {
                        o += type_size(p.count_type);
                        for (int64_t k = 0; k < p.list_len; k++, o += type_size(p.type))
                            e.data[p.name].first.push_back(read_typed(&d[o], p.type, big));
                    } else {
                        e.data[p.name].first.push_back(read_typed(&d[o], p.type, big));
                        o += type_size(p.type);
                    }
                }
            }
            off += e.length * rowsize;
        }
    }

    // ---- _elements_to_kwargs ------------------------------------------------------------------
    Element* ve = nullptr;
    Element* fe = nullptr;
    for (auto& e : els) {
        if (e.name == "vertex") ve = &e;
        if (e.name == "face") fe = &e;
    }
    if (!ve || ve->length == 0) throw std::runtime_error("No triangle mesh found in the file.");
    Geometry g;
    for (int64_t i = 0; i < ve->length; i++)
        for (const char* k : {"x", "y", "z"}) g.vertices.push_back(ve->data.at(k).first.at(i));
    bool has_faces = false;
    if (fe && fe->length) {
        std::string name;
        for (const char* n : {"vertex_index", "vertex_indices"})
            if (fe->data.count(n) || fe->ragged.count(n)) {
                name = n;
                break;
            }
        if (name.empty() && fe->props.size() == 1) name = fe->props[0].name;
        for (auto& p : fe->props)
            if (p.name == "texcoord") throw std::runtime_error("unsupported: PLY with per-face texcoords");
        std::vector<VecI> polys;
        if (fe->data.count(name)) {
            auto& fd = fe->data[name];
            int64_t w = fd.second;
            for (size_t i = 0; i + w <= fd.first.size(); i += w) {
                VecI p;
                for (int64_t k = 0; k < w; k++) p.push_back((int64_t)fd.first[i + k]);
                polys.push_back(p);
            }
        } else if (fe->ragged.count(name)) {
            for (auto& r : fe->ragged[name]) {
                VecI p;
                for (double x : r) p.push_back((int64_t)x);
                polys.push_back(p);
            }
        }
        g.faces = triangulate_quads(polys);
        has_faces = true;
    }
    for (const auto& names : std::vector<std::pair<std::string, std::string>>{{"texture_u", "texture_v"}, {"u", "v"}, {"s", "t"}})
        if (ve->data.count(names.first) && ve->data.count(names.second) && has_faces)
            throw std::runtime_error("unsupported: PLY with vertex texcoords");
    // colours: vertex colours override face colours (ColorVisuals setter order)
    auto colors = [&](Element* e, int& cols) -> VecD {
        VecD out;
        std::vector<const char*> keys;
        for (const char* k : {"red", "green", "blue", "alpha"})
            for (auto& p : e->props)
                if (p.name == k) keys.push_back(k);
        cols = (int)keys.size();
        if (keys.size() < 3) return out;
        bool isfloat = false;
        for (auto& p : e->props)
            for (auto k : keys)
                if (p.name == k && !is_int_type(p.type)) isfloat = true;
        for (int64_t i = 0; i < e->length; i++)
            for (auto k : keys) out.push_back(e->data.at(k).first.at(i));
        if (isfloat) {
            for (auto& x : out) x = (double)to_u8(x);
        } else {
            for (auto& x : out) x = (double)(uint8_t)(int64_t)x;
        }
        VecD rgba;
        for (int64_t i = 0; i < e->length; i++) {
            for (int k = 0; k < (int)keys.size(); k++) rgba.push_back(out[i * keys.size() + k]);
            if (keys.size() == 3) rgba.push_back(255.0);
        }
        cols = 4;
        return rgba;
    };
    int vcols = 0, fcols = 0;
    VecD vcol = colors(ve, vcols);
    VecD fcol = fe ? colors(fe, fcols) : VecD();
    if (!vcol.empty() && (int64_t)vcol.size() / 4 == ve->length) {
        g.visual.color_kind = 1;
        g.visual.color_cols = 4;
        g.visual.colors = vcol;
    } else if (!fcol.empty() && fe && (int64_t)fcol.size() / 4 == (int64_t)g.faces.size() / 3) {
        g.visual.color_kind = 2;
        g.visual.color_cols = 4;
        g.visual.colors = fcol;
    } else if (!fcol.empty()) {
        throw std::runtime_error("unsupported: PLY face colours on non-triangle faces");
    }
    LoadedScene S;
    S.geometries.push_back(std::move(g));
    SceneNode n;
    for (int i = 0; i < 16; i++) n.T[i] = (i % 5 == 0) ? 1.0 : 0.0;
    n.forder = false;
    n.geometry = 0;
    S.nodes.push_back(n);
    return S;
}

}  // namespace faqem
