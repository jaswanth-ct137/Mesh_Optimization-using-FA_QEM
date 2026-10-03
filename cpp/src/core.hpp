// FA-QEM geometric simplification (port of faqem/core.py).
#pragma once
#include <limits>
#include <map>
#include <string>

#include "bvh.hpp"
#include "compat/npcompat.hpp"

namespace faqem {

struct Options {
    // Table 1 weights ...
    double w_area = 100.0;
    double w_boundary = 500.0;
    double w_normal = 0.01;
    // ... except the two terms that hurt on real meshes (see README)
    double w_plane_area = 0.0;
    bool virtual_edges = false;
    double virtual_tau = 0.01;
    double min_edge_rel = 1e-8;
    double flip_threshold = 0.0;
    bool kappa_dimensionless = true;
    double boundary_window = 1;
    // implementation safeguards
    bool preserve_topology = true;
    double topology_penalty = 10.0;
    double max_move_factor = 2.0;
    double plane_area_clamp = 100.0;
    double max_rebuilds = 8;
    // automatic mode
    double max_error = 0.0;
    bool auto_mode = false;
    double local_tol = 0.0;

    // set a field by its Python name (DEFAULTS key); returns false if unknown
    bool set(const std::string& key, double value);
    static std::vector<std::string> keys();
    double get(const std::string& key) const;
    static bool is_bool(const std::string& key);
};

Options preset(const std::string& name);  // "Recommended", "Paper Table 1 (exact)", "Plain QEM (ablation baseline)"

struct SimplifyResult {
    VecD positions;  // (n, 3)
    VecI faces;      // (m, 3)
    VecI vertex_map; // (nV_in,)
    VecD history;    // (collapses, 7)
    std::map<std::string, double> stats;
    i64 output_faces() const { return (i64)faces.size() / 3; }
};

// reference: BVH of (P, F), reused by the local guarantee (built when null)
SimplifyResult simplify(const VecD& P, const VecI& F, i64 target_faces, const Options& options,
                        const BVH* reference = nullptr);

// exposed for parity tests
struct Quadrics {
    VecD Q, Qgeo, Ageo;
    double mean_area, mean_edge;
    i64 n_boundary_vertices;
};
Quadrics build_quadrics(const VecD& P, const VecI& F, const VecI& edges, const VecI& edge_counts, const Options& o);
bool solve_q(const double* q, double mx, double my, double mz, double* out, double rel_eps, double* Am, double* Vm);

}  // namespace faqem
