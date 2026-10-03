// STL and OFF loading with trimesh 5.1.0 semantics (process=False).
#include <cstring>
#include <stdexcept>

#include "io/asset.hpp"

namespace faqem {

namespace {

std::string lower(std::string s) {
    for (auto& c : s) c = (char)tolower((unsigned char)c);
    return s;
}
std::string strip(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\n\r\f\v");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\n\r\f\v");
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
// np.fromstring(text, sep=" ", dtype=float64): parse until the first invalid token
VecD fromstring(const std::string& text) {
    VecD out;
    for (const auto& t : split_ws(text)) {
        char* end = nullptr;
        double v = std::strtod(t.c_str(), &end);
        if (end == t.c_str() || *end != '\0') break;
        out.push_back(v);
    }
    return out;
}

LoadedScene single(Geometry g) {
    LoadedScene S;
    S.geometries.push_back(std::move(g));
    SceneNode n;
    for (int i = 0; i < 16; i++) n.T[i] = (i % 5 == 0) ? 1.0 : 0.0;
    n.forder = false;
    n.geometry = 0;
    S.nodes.push_back(n);
    return S;
}

// trimesh.util.comment_strip (reproduced literally, including how it re-joins the lead chunk)
std::string comment_strip(const std::string& text, const std::string& starts_with = "#",
                          const std::string& new_line = "\n") {
    if (text.find(starts_with) == std::string::npos) return text;
    std::string t = text + new_line;
    std::vector<std::string> split;
    size_t p = 0;
    while (true) {
        size_t q = t.find(starts_with, p);
        if (q == std::string::npos) {
            split.push_back(t.substr(p));
            break;
        }
        split.push_back(t.substr(p, q - p));
        p = q + starts_with.size();
    }
    std::string lead = text.rfind(starts_with, 0) == 0 ? "" : split[0];
    std::string result = lead + new_line;
    bool first = true;
    for (auto& s : split) {
        size_t nl = s.find(new_line);
        if (nl == std::string::npos) continue;
        std::string rest = s.substr(nl + new_line.size());
        if (rest.empty()) continue;
        if (!first) result += new_line;
        result += rest;
        first = false;
    }
    return strip(result);
}

}  // namespace

// trimesh.geometry.triangulate_quads for ragged / mixed polygons
VecI triangulate_quads(const std::vector<VecI>& polys) {
    VecI out;
    if (polys.empty()) return out;
    bool all3 = true, all4 = true;
    for (auto& p : polys) {
        all3 = all3 && p.size() == 3;
        all4 = all4 && p.size() == 4;
    }
    if (all3) {
        for (auto& p : polys) out.insert(out.end(), p.begin(), p.end());
        return out;
    }
    if (all4) {
        for (auto& p : polys) out.insert(out.end(), {p[0], p[1], p[2]});
        for (auto& p : polys) out.insert(out.end(), {p[2], p[3], p[0]});
        return out;
    }
    for (auto& p : polys)
        if (p.size() == 3) out.insert(out.end(), p.begin(), p.end());
    for (auto& p : polys)
        if (p.size() == 4) out.insert(out.end(), {p[0], p[1], p[2]});
    for (auto& p : polys)
        if (p.size() == 4) out.insert(out.end(), {p[2], p[3], p[0]});
    for (auto& p : polys)
        if (p.size() > 4)
            for (size_t i = 0; i + 2 < p.size(); i++) out.insert(out.end(), {p[0], p[i + 1], p[i + 2]});
    return out;
}

LoadedScene load_stl_scene(const std::string& path) {
    std::vector<uint8_t> d = read_file(path);
    // binary first (header + exact length), then ASCII
    if (d.size() >= 84) {
        uint32_t fc;
        std::memcpy(&fc, &d[80], 4);
        if (d.size() - 84 == (size_t)fc * 50) {
            Geometry g;
            if (fc == 0) return LoadedScene();
            g.vertices.resize((size_t)fc * 9);
            for (uint32_t f = 0; f < fc; f++) {
                const uint8_t* rec = &d[84 + (size_t)f * 50 + 12];
                for (int k = 0; k < 9; k++) {
                    float v;
                    std::memcpy(&v, rec + 4 * k, 4);
                    g.vertices[(size_t)f * 9 + k] = (double)v;
                }
            }
            g.faces.resize((size_t)fc * 3);
            for (size_t i = 0; i < g.faces.size(); i++) g.faces[i] = (int64_t)i;
            return single(std::move(g));
        }
    }
    std::string raw_mixed = strip(std::string(d.begin(), d.end()));
    std::string raw_lower = lower(raw_mixed);
    LoadedScene S;
    size_t position = 0;
    for (size_t it = 0; it < raw_mixed.size(); it++) {
        size_t ss = raw_lower.find("solid", position);
        size_t se = raw_lower.find("endsolid", position);
        if (se == std::string::npos || ss == std::string::npos) break;
        position = se + 8;
        if (ss > se) throw std::runtime_error("`endsolid` precedes `solid`!");
        std::string solid = raw_lower.substr(ss, se - ss);
        std::string joined;
        size_t p = solid.find("vertex");
        bool firstv = true;
        while (p != std::string::npos) {
            size_t s = p + 6;
            size_t q = solid.find("vertex", s);
            std::string chunk = solid.substr(s, (q == std::string::npos ? solid.size() : q) - s);
            // line[: line.find("\n")] (find == -1 drops the last character)
            size_t nl = chunk.find('\n');
            chunk = nl == std::string::npos ? chunk.substr(0, chunk.empty() ? 0 : chunk.size() - 1) : chunk.substr(0, nl);
            if (!firstv) joined += " ";
            joined += chunk;
            firstv = false;
            p = q;
        }
        VecD v = fromstring(joined);
        if (v.size() < 3) continue;
        if (v.size() % 3 != 0) throw std::runtime_error("incorrect number of vertices");
        Geometry g;
        g.vertices = v;
        size_t nv = v.size() / 3;
        g.faces.resize((nv / 3) * 3);
        for (size_t i = 0; i < g.faces.size(); i++) g.faces[i] = (int64_t)i;
        S.geometries.push_back(std::move(g));
        SceneNode n;
        for (int i = 0; i < 16; i++) n.T[i] = (i % 5 == 0) ? 1.0 : 0.0;
        n.forder = false;
        n.geometry = (int)S.geometries.size() - 1;
        S.nodes.push_back(n);
    }
    return S;
}

LoadedScene load_off_scene(const std::string& path) {
    std::vector<uint8_t> d = read_file(path);
    std::string text = strip(comment_strip(std::string(d.begin(), d.end())));
    size_t h1 = text.find("COFF"), h2 = text.find("OFF");
    size_t h = std::min(h1, h2);
    if (h == std::string::npos) throw std::runtime_error("Not an OFF file!");
    std::string header = h == h1 ? "COFF" : "OFF";
    std::string raw = text.substr(h + header.size());
    std::vector<std::string> lines;
    size_t p = 0;
    while (p <= raw.size()) {
        size_t q = raw.find_first_of("\n\r", p);
        std::string ln = strip(raw.substr(p, q == std::string::npos ? std::string::npos : q - p));
        if (!ln.empty()) lines.push_back(ln);
        if (q == std::string::npos) break;
        p = q + 1;
        if (raw[q] == '\r' && p < raw.size() && raw[p] == '\n') p++;
    }
    if (lines.empty()) throw std::runtime_error("OFF file is missing the vertex/face count line");
    auto hd = split_ws(lines[0]);
    if (hd.size() < 2) throw std::runtime_error("OFF file has a malformed vertex/face count line");
    int64_t nv = std::stoll(hd[0]), nf = std::stoll(hd[1]);
    Geometry g;
    for (int64_t i = 1; i <= nv; i++) {
        auto t = split_ws(lines.at(i));
        for (int k = 0; k < 3; k++) g.vertices.push_back(std::strtod(t.at(k).c_str(), nullptr));
    }
    std::vector<VecI> polys;
    for (int64_t i = nv + 1; i < nv + nf + 1 && i < (int64_t)lines.size(); i++) {
        auto t = split_ws(lines[i]);
        int64_t c = std::stoll(t[0]);
        VecI poly;
        for (int64_t k = 1; k < c + 1 && k < (int64_t)t.size(); k++) poly.push_back(std::stoll(t[k]));
        polys.push_back(poly);
    }
    // trimesh keeps the face tokens as strings: a ragged face list with 5+-gons fails in
    // triangle_fans_to_faces (string * array), so the file cannot be loaded
    bool ragged = false, big = false;
    for (auto& q : polys) {
        ragged = ragged || q.size() != polys[0].size();
        big = big || q.size() > 4;
    }
    if (ragged && big) throw std::runtime_error("OFF: mixed polygons with more than 4 corners are not loadable");
    g.faces = triangulate_quads(polys);
    return single(std::move(g));
}

}  // namespace faqem
