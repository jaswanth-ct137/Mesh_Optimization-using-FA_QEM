#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct Vertex {
  std::array<double, 3> position{};
};

struct MeshPart {
  std::string name;
  std::vector<Vertex> vertices;
  std::vector<std::uint32_t> indices;
  std::array<float, 4> color{0.72f, 0.76f, 0.82f, 1.0f};
};

struct MeshScene {
  std::vector<MeshPart> parts;
};

struct DisplayStats {
  std::size_t vertices = 0;
  std::size_t triangles = 0;
};

inline DisplayStats display_stats(const MeshScene &scene) {
  DisplayStats result;
  for (const auto &part : scene.parts) {
    result.vertices += part.vertices.size();
    result.triangles += part.indices.size() / 3;
  }
  return result;
}
