// Automatic (best face count) and best-result-for-a-face-count searches
// (port of faqem/pipeline.py: simplify_auto, simplify_to_count).
#pragma once
#include <functional>
#include <map>
#include <string>

#include "core.hpp"
#include "metrics.hpp"
#include "prep.hpp"

namespace faqem {

// Automatic mode: allowed worst-case deviation, fraction of the bbox diagonal
const std::map<std::string, double>& detail_levels();

struct SearchPass {
    double factor_or_tau;  // simplify_auto: local factor; simplify_to_count: tau
    i64 faces;
    double hausdorff;      // NaN when not measured
    bool ok;               // simplify_auto: within the limit; simplify_to_count: reached the count
};

struct SearchResult {
    SimplifyResult res;
    Metrics metrics;
    std::vector<SearchPass> passes;
    bool within = false;          // simplify_auto: a pass was within the limit
    double plain_hausdorff = 0;   // simplify_to_count: the plain FA-QEM pass
    double time_total = 0;
};

// called after each pass: (pass index, faces of that pass)
using PassProgress = std::function<void(int, i64)>;

// prep + bvh of the prepared mesh (bvh built lazily by the caller)
SearchResult simplify_auto(const PreparedMesh& prep, const BVH& bvh, double tolerance, const Options& options,
                           int max_passes = 4, const PassProgress& progress = nullptr);
SearchResult simplify_to_count(const PreparedMesh& prep, const BVH& bvh, i64 target_faces, const Options& options,
                               int max_passes = 6, const PassProgress& progress = nullptr);

}  // namespace faqem
