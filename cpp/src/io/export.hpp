// Mesh export with trimesh 5.1.0 exporter semantics (port of faqem/io.py export_* and
// faqem/wireframe.py): identical accessor arrays (float32 positions, unitized normals, flipped
// uv, uint32 indices) and material values; the .obj text is byte-identical.
#pragma once
#include <string>

#include "io/image.hpp"
#include "compat/npcompat.hpp"

namespace faqem {

// trimesh.Trimesh(P, F, process=False).vertex_normals (angle-weighted, unitized)
VecD trimesh_vertex_normals(const VecD& P, const VecI& F);

// io.export_geometry: .obj (text) or .glb (grey PBR material, optional normals)
std::string export_geometry(const std::string& path, const VecD& positions, const VecI& faces,
                            const VecD* normals = nullptr, const double color[4] = nullptr);

// io.export_textured: unshared-corner mesh with baked PBR textures -> GLB
std::string export_textured(const std::string& path, const VecD& positions, const VecD& normals, const VecD& uv,
                            const Image* color_img, const Image* normal_img, const Image* mr_img,
                            const double base_color[4], double metallic, double roughness, bool double_sided,
                            const std::string& alpha_mode);

// wireframe.add_wireframe: glb_in empty -> lines only
std::string add_wireframe(const std::string& glb_in, const VecD& P, const VecI& F, const std::string& glb_out,
                          const double color[4], double lift = 2e-4, double alpha = -1.0);

}  // namespace faqem
