// Mesh pre-processing (port of faqem/prep.py).
#pragma once
#include <map>
#include <string>

#include "compat/npcompat.hpp"

namespace faqem {

struct PreparedMesh {
    VecD positions;        // (n, 3) normalised space
    VecI faces;            // (m, 3)
    VecI face_orig;        // (m,) prepared face -> input face
    VecI input_to_vertex;  // (n_in,) input vertex -> prepared vertex (-1 if unused)
    double center[3] = {0, 0, 0};
    double diag = 1.0;
    std::map<std::string, i64> removed;
    i64 nV() const { return (i64)positions.size() / 3; }
    i64 nF() const { return (i64)faces.size() / 3; }
    // p * diag + center
    VecD to_model(const VecD& p) const;
};

PreparedMesh prepare_mesh(const VecD& positions, const VecI& faces, double weld_tolerance = 1e-6,
                          bool make_manifold = true);

// unique undirected edges (e, 2) sorted lexicographically, and their incident face counts
void unique_edges(const VecI& F, i64 nV, VecI& edges, VecI& counts);

struct MeshStats {
    i64 vertices, faces, edges, components, boundary_edges, non_manifold_edges;
};
MeshStats mesh_stats(i64 nV, const VecI& F);

}  // namespace faqem
