// faqem: command-line FA-QEM simplification (same flags and report as cli.py).
//
//     faqem input.glb --ratio 0.1 -o out_dir
//     faqem scan.stl --faces 5000 --no-normal-map
//     faqem doll.glb --auto Medium           # face count chosen for a max deviation
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "compat/fmath.hpp"
#include "digest.hpp"
#include "parallel.hpp"
#include "pipeline.hpp"

using namespace faqem;

// Python repr(float)
static std::string pyrepr(double x) {
    if (std::isnan(x)) return "nan";
    if (std::isinf(x)) return x > 0 ? "inf" : "-inf";
    if (x == 0) return std::signbit(x) ? "-0.0" : "0.0";
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof buf, x, std::chars_format::scientific);
    std::string s(buf, r.ptr);
    size_t e = s.find('e');
    std::string mant = s.substr(0, e);
    int exp = std::stoi(s.substr(e + 1));
    bool neg = mant[0] == '-';
    if (neg) mant = mant.substr(1);
    std::string digits;
    for (char c : mant)
        if (c != '.') digits += c;
    std::string out;
    if (exp < -4 || exp >= 16) {
        out = digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        char eb[16];
        std::snprintf(eb, sizeof eb, "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
        out += eb;
    } else if (exp < 0) {
        out = "0." + std::string(-exp - 1, '0') + digits;
    } else {
        if ((int)digits.size() <= exp + 1) out = digits + std::string(exp + 1 - digits.size(), '0') + ".0";
        else out = digits.substr(0, exp + 1) + "." + digits.substr(exp + 1);
    }
    return (neg ? "-" : "") + out;
}

static double pyround(double x, int nd) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*f", nd, x);
    return std::strtod(buf, nullptr);
}

static void usage() {
    std::fprintf(stderr,
                 "usage: faqem input [--ratio R | --faces N | --auto {Low,Medium,High,Ultra} | --auto-deviation PCT]\n"
                 "             [--fast] [-o OUT] [--no-color] [--no-normal-map] [--atlas N] [--no-metrics] [--digest]\n"
                 "             [--threads N]  (default: all cores; results are identical for any N)\n"
                 "             [--w-area X] [--w-boundary X] ... (any FA-QEM option, see core.DEFAULTS)\n");
}

int main(int argc, char** argv) {
    std::string input, out = "output", auto_level;
    double ratio = 0.1, auto_dev = 0;
    long long faces = 0;
    bool fast = false, no_color = false, no_normal = false, no_metrics = false, digest = false;
    int atlas = 0;
    int exclusive = 0;
    Options opts;
    static const std::vector<std::string> internal = {"auto", "max_error", "local_tol"};
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--ratio") {
            ratio = std::stod(next());
            exclusive++;
        } else if (a == "--faces") {
            faces = std::stoll(next());
            exclusive++;
        } else if (a == "--auto") {
            auto_level = next();
            if (!detail_levels().count(auto_level)) {
                usage();
                return 2;
            }
            exclusive++;
        } else if (a == "--auto-deviation") {
            auto_dev = std::stod(next());
            exclusive++;
        } else if (a == "--fast") fast = true;
        else if (a == "-o" || a == "--out") out = next();
        else if (a == "--no-color") no_color = true;
        else if (a == "--no-normal-map") no_normal = true;
        else if (a == "--atlas") atlas = std::stoi(next());
        else if (a == "--no-metrics") no_metrics = true;
        else if (a == "--digest") digest = true;
        else if (a == "--threads") set_num_threads(std::stoi(next()));
        else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else if (a.rfind("--", 0) == 0) {
            std::string key = a.substr(2);
            for (auto& c : key)
                if (c == '-') c = '_';
            bool known = false;
            for (auto& k : Options::keys())
                if (k == key && std::find(internal.begin(), internal.end(), k) == internal.end()) known = true;
            if (!known) {
                std::fprintf(stderr, "unrecognized argument: %s\n", a.c_str());
                return 2;
            }
            std::string v = next();
            if (Options::is_bool(key)) {
                std::string lv = v;
                for (auto& c : lv) c = (char)tolower(c);
                opts.set(key, (lv == "1" || lv == "true" || lv == "yes") ? 1.0 : 0.0);
            } else {
                opts.set(key, std::stod(v));
            }
        } else if (input.empty()) {
            input = a;
        } else {
            usage();
            return 2;
        }
    }
    if (input.empty() || exclusive > 1) {
        usage();
        return 2;
    }
    try {
        auto t0 = std::chrono::steady_clock::now();
        LoadedMesh mesh(input);
        double lt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const MeshStats& s = mesh.stats;
        std::printf("loaded %s: {'vertices': %lld, 'faces': %lld, 'edges': %lld, 'components': %lld, "
                    "'boundary_edges': %lld, 'non_manifold_edges': %lld} (%.1fs)\n",
                    input.c_str(), (long long)s.vertices, (long long)s.faces, (long long)s.edges,
                    (long long)s.components, (long long)s.boundary_edges, (long long)s.non_manifold_edges, lt);
        std::printf("engine: C++ (%s build, %d threads)\n", fm::build_mode(), num_threads());
        RunOptions ro;
        ro.options = opts;
        ro.target_faces = faces ? faces : (long long)std::nearbyint((double)s.faces * ratio);
        ro.auto_tolerance = auto_dev ? auto_dev / 100.0 : (auto_level.empty() ? 0.0 : detail_levels().at(auto_level));
        ro.fast = fast;
        ro.bake_color = !no_color;
        ro.bake_normal = !no_normal;
        ro.atlas_size = atlas;
        ro.compute_metrics = !no_metrics;
        RunOutput r = run(mesh, out, ro, [](double f, const std::string& msg) {
            std::printf("  [%5.1f%%] %s\n", f * 100, msg.c_str());
            std::fflush(stdout);
        });
        auto& st = r.res.stats;
        std::printf("\n%s -> %s faces in %.2fs (bake %.2fs)\n",
                    [](long long v) {
                        std::string s = std::to_string(v), o;
                        for (size_t i = 0; i < s.size(); i++) {
                            if (i && (s.size() - i) % 3 == 0) o += ',';
                            o += s[i];
                        }
                        return o;
                    }((long long)st["input_faces"]).c_str(),
                    [](long long v) {
                        std::string s = std::to_string(v), o;
                        for (size_t i = 0; i < s.size(); i++) {
                            if (i && (s.size() - i) % 3 == 0) o += ',';
                            o += s[i];
                        }
                        return o;
                    }((long long)st["output_faces"]).c_str(),
                    st["time_total"], r.time_bake);
        if (r.best_for_count)
            std::printf("best for this face count: worst deviation %.3f%% (plain FA-QEM %.3f%%), %zu passes\n",
                        r.metrics.hausdorff * 100, r.plain_hausdorff * 100, r.passes.size());
        if (r.has_metrics)
            std::printf("fidelity: {\"hausdorff\": %s, \"chamfer\": %s, \"mean_distance\": %s}\n",
                        pyrepr(pyround(r.metrics.hausdorff, 7)).c_str(), pyrepr(pyround(r.metrics.chamfer, 7)).c_str(),
                        pyrepr(pyround(r.metrics.mean_distance, 7)).c_str());
        auto absp = [](const std::string& p) { return std::filesystem::absolute(p).lexically_normal().string(); };
        if (!r.textured_glb.empty()) std::printf("  %-13s %s\n", "textured_glb", absp(r.textured_glb).c_str());
        if (digest) {
            // SHA-256 of the result arrays (raw little-endian bits) and of every output file
            auto hv = [](const auto& v) { return sha256_hex(v.data(), v.size() * sizeof(v[0])); };
            std::printf("digest mode %s\n", fm::build_mode());
            std::printf("digest positions %s\n", hv(r.res.positions).c_str());
            std::printf("digest faces %s\n", hv(r.res.faces).c_str());
            std::printf("digest vertex_map %s\n", hv(r.res.vertex_map).c_str());
            Sha256 ps;
            for (auto& p : r.passes) {
                ps.update(&p.factor_or_tau, 8);
                ps.update(&p.faces, 8);
                ps.update(&p.hausdorff, 8);
                char ok = p.ok;
                ps.update(&ok, 1);
            }
            std::printf("digest passes %s\n", ps.hex().c_str());
            double mv[3] = {r.metrics.hausdorff, r.metrics.chamfer, r.metrics.mean_distance};
            std::printf("digest metrics %s\n", sha256_hex(mv, sizeof mv).c_str());
            for (const std::string* f : {&r.geometry_obj, &r.geometry_glb, &r.textured_glb, &r.clay_wire_glb,
                                         &r.lines_glb, &r.textured_wire_glb})
                if (!f->empty())
                    std::printf("digest file %s %s\n", std::filesystem::path(*f).filename().string().c_str(),
                                sha256_file(*f).c_str());
        }
        std::printf("  %-13s %s\n", "geometry_glb", absp(r.geometry_glb).c_str());
        std::printf("  %-13s %s\n", "geometry_obj", absp(r.geometry_obj).c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
