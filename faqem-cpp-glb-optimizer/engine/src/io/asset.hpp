// Loading meshes with their appearance (port of faqem/io.py + the trimesh loaders it relies on).
#pragma once
#include <map>
#include <string>
#include <vector>

#include "compat/npcompat.hpp"
#include "io/image.hpp"

namespace faqem {

struct Material {
    double color[4] = {0.8, 0.8, 0.8, 1.0};  // linear RGBA factor
    ImagePtr texture;                          // RGBA uint8 sRGB, or null
    double metallic = 0.0;
    double roughness = 1.0;
    ImagePtr mr_texture;  // glTF: G = roughness, B = metallic
    std::string alpha_mode = "OPAQUE";
    bool double_sided = false;
};

struct Asset {
    VecD positions;   // (n, 3) model space
    VecI faces;       // (m, 3)
    VecI face_material;
    std::vector<Material> materials;
    bool has_uv = false;
    VecD corner_uv;   // (m, 3, 2) OpenGL convention
    bool has_col = false;
    VecD corner_color;  // (m, 3, 4) linear RGBA
    std::map<std::string, double> info;

    bool has_texture() const;
    bool has_appearance() const;
    bool has_mr_variation() const;
};

// load any supported mesh file (glb, gltf, obj, stl, ply, off) into a flat Asset
Asset load_asset(const std::string& path);

// trimesh-level intermediate: one geometry per scene node, in trimesh's nodes_geometry order
struct Visual {
    // TextureVisuals (material set) or ColorVisuals
    bool texture_visuals = false;
    int material_key = -1;  // identity of the material object (-1: None)
    bool has_uv = false;
    VecD uv;                // (n, 2)
    // vertex colours: ColorVisuals vertex_colors (uint8 RGBA) or TextureVisuals vertex_attributes['color']
    int color_kind = 0;     // 0 none, 1 ColorVisuals vertex, 2 ColorVisuals face, 3 vertex_attributes
    VecD colors;            // raw values as float64
    int color_cols = 0;
};
struct Geometry {
    VecD vertices;  // (n, 3)
    VecI faces;     // (m, 3)
    Visual visual;
};
struct SceneNode {
    double T[16];      // model transform (row-major 4x4)
    bool forder;       // numpy layout of that transform
    int geometry;      // index into geometries
};
struct LoadedScene {
    std::vector<Geometry> geometries;
    std::vector<SceneNode> nodes;
    std::vector<Material> materials;  // by material key
};

LoadedScene load_gltf_scene(const std::string& path);
LoadedScene load_stl_scene(const std::string& path);
LoadedScene load_off_scene(const std::string& path);
LoadedScene load_ply_scene(const std::string& path);
LoadedScene load_obj_scene(const std::string& path);

// trimesh.visual.color.to_rgba for a float32 (n, k) array -> uint8 RGBA (as double values)
VecD to_rgba_f32(const std::vector<float>& c, int cols);

std::vector<uint8_t> read_file(const std::string& path);

}  // namespace faqem
