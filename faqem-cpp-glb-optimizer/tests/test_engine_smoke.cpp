// FA-QEM engine smoke test: prepare + plain simplify + Automatic search on a subdivided octahedron.
#include "engine_bridge.hpp"

#include <cassert>
#include <map>
#include <cmath>
#include <iostream>

namespace {
// octahedron, each face split into 4 twice and projected to the unit sphere (128 faces)
void sphere(faqem::VecD &P, faqem::VecI &F) {
  P = {1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1};
  F = {0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5};
  for (int level = 0; level < 2; ++level) {
    faqem::VecI next;
    std::map<std::pair<long long, long long>, long long> mid;
    auto midpoint = [&](long long a, long long b) {
      auto key = std::make_pair(std::min(a, b), std::max(a, b));
      auto it = mid.find(key);
      if (it != mid.end())
        return it->second;
      double m[3], n = 0;
      for (int c = 0; c < 3; ++c) {
        m[c] = (P[3 * a + c] + P[3 * b + c]) / 2;
        n += m[c] * m[c];
      }
      for (int c = 0; c < 3; ++c)
        P.push_back(m[c] / std::sqrt(n));
      return mid[key] = static_cast<long long>(P.size() / 3 - 1);
    };
    for (std::size_t f = 0; f < F.size(); f += 3) {
      long long a = F[f], b = F[f + 1], c = F[f + 2];
      long long ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
      for (long long t : {a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca})
        next.push_back(t);
    }
    F = next;
  }
}
}  // namespace

int main() {
  faqem::VecD P;
  faqem::VecI F;
  sphere(P, F);
  assert(F.size() / 3 == 128);
  const auto prep = faqem::prepare_mesh(P, F);
  assert(prep.nF() == 128);

  const auto plain = faqem::simplify(prep.positions, prep.faces, 32, faqem::Options());
  assert(plain.output_faces() > 0 && plain.output_faces() <= 32);
  for (auto v : plain.faces)
    assert(v >= 0 && v < static_cast<long long>(plain.positions.size() / 3));

  // Automatic on a flat 20 x 20 grid (800 faces): the deviation limit allows a large reduction
  faqem::VecD GP;
  faqem::VecI GF;
  const int n = 20;
  for (int y = 0; y <= n; ++y)
    for (int x = 0; x <= n; ++x)
      GP.insert(GP.end(), {double(x), double(y), 0.0});
  for (int y = 0; y < n; ++y)
    for (int x = 0; x < n; ++x) {
      const long long v = y * (n + 1) + x;
      GF.insert(GF.end(), {v, v + 1, v + n + 2, v, v + n + 2, v + n + 1});
    }
  const auto grid = faqem::prepare_mesh(GP, GF);
  faqem::BVH bvh(grid.positions, grid.faces);
  const auto automatic = faqem::simplify_auto(grid, bvh, detail_tolerance("Low"), faqem::Options());
  assert(automatic.res.output_faces() > 0 && automatic.res.output_faces() < 400);
  assert(automatic.within);
  assert(automatic.metrics.hausdorff <= detail_tolerance("Low"));
  assert(!automatic.passes.empty());
  assert(std::isfinite(automatic.metrics.hausdorff));
  std::cout << "engine smoke test passed: plain " << plain.output_faces() << " faces, automatic "
            << automatic.res.output_faces() << " faces, Hausdorff " << automatic.metrics.hausdorff
            << "\n";
}
