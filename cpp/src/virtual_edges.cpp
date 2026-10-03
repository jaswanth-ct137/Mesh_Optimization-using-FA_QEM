// Port of faqem/virtual_edges.py. KD-tree queries use scipy 1.15.0's own cKDTree sources
// (third_party/ckdtree) built and queried exactly as scipy.spatial.cKDTree(data).query(x,
// distance_upper_bound=tau) does, so neighbour ties resolve identically.
#include "virtual_edges.hpp"

#include <limits>
#include <memory>

#include "ckdtree_decl.h"
#include "compat/csgraph.hpp"

namespace faqem {

namespace {

class KDTree {
public:
    explicit KDTree(const VecD& pts) : data_(pts) {
        i64 n = (i64)data_.size() / 3;
        indices_.resize(n);
        for (i64 i = 0; i < n; i++) indices_[i] = (ckdtree_intp_t)i;
        maxes_.assign(3, 0.0);
        mins_.assign(3, 0.0);
        if (n > 0)
            for (int k = 0; k < 3; k++) {
                maxes_[k] = mins_[k] = data_[k];
                for (i64 i = 1; i < n; i++) {
                    maxes_[k] = np::fmax0(maxes_[k], data_[i * 3 + k]);
                    mins_[k] = np::fmin0(mins_[k], data_[i * 3 + k]);
                }
            }
        buffer_.reset(new std::vector<ckdtreenode>());
        t_.tree_buffer = buffer_.get();
        t_.raw_data = data_.data();
        t_.n = n;
        t_.m = 3;
        t_.leafsize = 16;
        t_.raw_maxes = maxes_.data();
        t_.raw_mins = mins_.data();
        t_.raw_indices = indices_.data();
        t_.raw_boxsize_data = nullptr;
        std::vector<double> tmpmaxes = maxes_, tmpmins = mins_;
        build_ckdtree(&t_, 0, n, tmpmaxes.data(), tmpmins.data(), 1, 1);
        // _post_init: fix the child pointers after the buffer stopped growing
        t_.ctree = buffer_->data();
        t_.size = (ckdtree_intp_t)buffer_->size();
        for (auto& nd : *buffer_) {
            if (nd.split_dim == -1) {
                nd.less = nullptr;
                nd.greater = nullptr;
            } else {
                nd.less = t_.ctree + nd._less;
                nd.greater = t_.ctree + nd._greater;
            }
        }
    }
    // query(x, k=1, distance_upper_bound=ub): nearest index (n when none within the bound)
    void query(const VecD& x, double ub, VecD& d, VecI& idx) const {
        i64 nq = (i64)x.size() / 3;
        d.assign(nq, std::numeric_limits<double>::infinity());
        std::vector<ckdtree_intp_t> ii(nq, (ckdtree_intp_t)t_.n);
        ckdtree_intp_t k = 1;
        if (nq) query_knn(&t_, d.data(), ii.data(), x.data(), nq, &k, 1, 1, 0.0, 2.0, ub);
        idx.assign(ii.begin(), ii.end());
    }

private:
    VecD data_;
    std::vector<ckdtree_intp_t> indices_;
    VecD maxes_, mins_;
    std::unique_ptr<std::vector<ckdtreenode>> buffer_;
    ckdtree t_{};
};

}  // namespace

VecI find_virtual_edges(const VecD& P, const VecI& F, double tau, i64 max_components) {
    i64 nV = (i64)P.size() / 3, nF = (i64)F.size() / 3;
    // face_components: edges (0,1) and (1,2) of every face
    VecI src, dst;
    for (int e = 0; e < 2; e++)
        for (i64 f = 0; f < nF; f++) {
            src.push_back(F[f * 3 + e]);
            dst.push_back(F[f * 3 + e + 1]);
        }
    VecI labels;
    i64 n_comp = csg::connected_components(nV, src, dst, labels);
    if (n_comp <= 1) return {};
    VecI fcomp(nF);
    for (i64 f = 0; f < nF; f++) fcomp[f] = labels[F[f * 3]];
    VecD centroids(nF * 3);
    for (i64 f = 0; f < nF; f++)
        for (int k = 0; k < 3; k++) {
            double s = 0.0;
            for (int j = 0; j < 3; j++) s = s + P[F[f * 3 + j] * 3 + k];
            centroids[f * 3 + k] = s / 3;
        }
    VecI order = np::argsort_stable(fcomp);
    std::vector<VecI> comps(n_comp);
    for (i64 i : order) comps[fcomp[i]].push_back(i);
    VecI neglen(n_comp);
    for (i64 c = 0; c < n_comp; c++) neglen[c] = -(i64)comps[c].size();
    VecI comp_order = np::argsort_quick(neglen);
    if ((i64)comp_order.size() > max_components) comp_order.resize(max_components);
    i64 nc = (i64)comp_order.size();
    VecD lo(nc * 3), hi(nc * 3);
    for (i64 i = 0; i < nc; i++) {
        const VecI& fs = comps[comp_order[i]];
        for (int k = 0; k < 3; k++) {
            double mn = centroids[fs[0] * 3 + k], mx = mn;
            for (size_t j = 1; j < fs.size(); j++) {
                mn = np::fmin0(mn, centroids[fs[j] * 3 + k]);
                mx = np::fmax0(mx, centroids[fs[j] * 3 + k]);
            }
            lo[i * 3 + k] = mn - tau;
            hi[i * 3 + k] = mx + tau;
        }
    }
    VecD best_d(nV, std::numeric_limits<double>::infinity());
    VecI best_u(nV, -1);
    std::vector<std::unique_ptr<KDTree>> trees(n_comp);
    for (i64 ii = 0; ii < nc; ii++) {
        VecI partners;
        for (i64 jj = ii + 1; jj < nc; jj++) {
            bool ov = true;
            for (int k = 0; k < 3; k++) ov = ov && (lo[jj * 3 + k] <= hi[ii * 3 + k]) && (hi[jj * 3 + k] >= lo[ii * 3 + k]);
            if (ov) partners.push_back(jj);
        }
        if (partners.empty()) continue;
        i64 ci = comp_order[ii];
        if (!trees[ci]) {
            VecD pts;
            for (i64 f : comps[ci])
                for (int k = 0; k < 3; k++) pts.push_back(centroids[f * 3 + k]);
            trees[ci].reset(new KDTree(pts));
        }
        for (i64 jj : partners) {
            i64 cj = comp_order[jj];
            const VecI& fj = comps[cj];
            VecD q;
            for (i64 f : fj)
                for (int k = 0; k < 3; k++) q.push_back(centroids[f * 3 + k]);
            VecD d;
            VecI kidx;
            trees[ci]->query(q, tau, d, kidx);
            VecI fa, fb;
            for (size_t h = 0; h < fj.size(); h++)
                if (std::isfinite(d[h])) {
                    fa.push_back(comps[ci][kidx[h]]);
                    fb.push_back(fj[h]);
                }
            if (fa.empty()) continue;
            i64 nh = (i64)fa.size();
            VecI a(nh), b(nh);
            VecD dd(nh);
            for (i64 h = 0; h < nh; h++) {
                double bestv = 0;
                int arg = -1;
                for (int s = 0; s < 3; s++)
                    for (int t = 0; t < 3; t++) {
                        const double* pa = &P[F[fa[h] * 3 + s] * 3];
                        const double* pb = &P[F[fb[h] * 3 + t] * 3];
                        double dx = pa[0] - pb[0], dy = pa[1] - pb[1], dz = pa[2] - pb[2];
                        double v = (dx * dx + dy * dy) + dz * dz;
                        if (arg < 0 || v < bestv) {  // argmin: first minimum
                            bestv = v;
                            arg = s * 3 + t;
                        }
                    }
                a[h] = F[fa[h] * 3 + arg / 3];
                b[h] = F[fb[h] * 3 + arg % 3];
                dd[h] = bestv;
            }
            for (int pass = 0; pass < 2; pass++) {
                const VecI& x = pass == 0 ? a : b;
                const VecI& y = pass == 0 ? b : a;
                VecD neg(nh);
                for (i64 h = 0; h < nh; h++) neg[h] = -dd[h];
                VecI o = np::argsort_quick(neg);
                // better is evaluated against best_d before this batch; the last write wins
                std::vector<char> better(nh);
                for (i64 h = 0; h < nh; h++) better[h] = dd[o[h]] < best_d[x[o[h]]];
                for (i64 h = 0; h < nh; h++)
                    if (better[h]) {
                        best_d[x[o[h]]] = dd[o[h]];
                        best_u[x[o[h]]] = y[o[h]];
                    }
            }
        }
    }
    VecI rows;
    for (i64 v = 0; v < nV; v++)
        if (best_u[v] >= 0) {
            rows.push_back(std::min(v, best_u[v]));
            rows.push_back(std::max(v, best_u[v]));
        }
    if (rows.empty()) return {};
    np::UniqueRows u = np::unique_rows(rows.data(), (i64)rows.size() / 2, 2);
    VecI out;
    for (size_t i = 0; i < u.index.size(); i++)
        if (u.rows[i * 2] != u.rows[i * 2 + 1]) {
            out.push_back(u.rows[i * 2]);
            out.push_back(u.rows[i * 2 + 1]);
        }
    return out;
}

}  // namespace faqem
