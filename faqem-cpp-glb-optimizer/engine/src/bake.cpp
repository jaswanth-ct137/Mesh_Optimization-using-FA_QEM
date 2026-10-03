// Port of faqem/bake.py. The Numba kernels are transliterated; the prange over charts becomes a
// thread pool (each chart writes only its own atlas cell, so the result is order-independent).
#include "bake.hpp"

#include <limits>
#include <stdexcept>

#include "compat/fmath.hpp"
#include "compat/nprandom.hpp"
#include "metrics.hpp"

namespace faqem {

static const double INF = std::numeric_limits<double>::infinity();
static const int PAD = 2;
static const int GAP = 2;

void corner_normals(const VecD& P, const VecI& F, double crease_cos, VecD& out, VecD& fu) {
    i64 nV = (i64)P.size() / 3, nF = (i64)F.size() / 3;
    VecD fn(nF * 3, 0.0);
    fu.assign(nF * 3, 0.0);
    for (i64 f = 0; f < nF; f++) {
        i64 a = F[f * 3], b = F[f * 3 + 1], c = F[f * 3 + 2];
        double ux = P[b * 3] - P[a * 3], uy = P[b * 3 + 1] - P[a * 3 + 1], uz = P[b * 3 + 2] - P[a * 3 + 2];
        double wx = P[c * 3] - P[a * 3], wy = P[c * 3 + 1] - P[a * 3 + 1], wz = P[c * 3 + 2] - P[a * 3 + 2];
        double nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
        fn[f * 3] = nx;
        fn[f * 3 + 1] = ny;
        fn[f * 3 + 2] = nz;
        double l = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (l > 0) {
            fu[f * 3] = nx / l;
            fu[f * 3 + 1] = ny / l;
            fu[f * 3 + 2] = nz / l;
        }
    }
    VecI deg(nV + 1, 0);
    for (i64 f = 0; f < nF; f++)
        for (int k = 0; k < 3; k++) deg[F[f * 3 + k] + 1] += 1;
    for (i64 i = 0; i < nV; i++) deg[i + 1] += deg[i];
    VecI fill(deg.begin(), deg.begin() + nV);
    VecI vf(3 * nF);
    for (i64 f = 0; f < nF; f++)
        for (int k = 0; k < 3; k++) {
            i64 v = F[f * 3 + k];
            vf[fill[v]] = f;
            fill[v] += 1;
        }
    out.assign(nF * 9, 0.0);
    for (i64 f = 0; f < nF; f++)
        for (int k = 0; k < 3; k++) {
            i64 v = F[f * 3 + k];
            double sx = 0.0, sy = 0.0, sz = 0.0;
            for (i64 j = deg[v]; j < deg[v + 1]; j++) {
                i64 g = vf[j];
                if (fu[g * 3] * fu[f * 3] + fu[g * 3 + 1] * fu[f * 3 + 1] + fu[g * 3 + 2] * fu[f * 3 + 2] >= crease_cos) {
                    sx += fn[g * 3];
                    sy += fn[g * 3 + 1];
                    sz += fn[g * 3 + 2];
                }
            }
            double l = std::sqrt(sx * sx + sy * sy + sz * sz);
            double* o = &out[(f * 3 + k) * 3];
            if (l > 0) {
                o[0] = sx / l;
                o[1] = sy / l;
                o[2] = sz / l;
            } else {
                o[0] = fu[f * 3];
                o[1] = fu[f * 3 + 1];
                o[2] = fu[f * 3 + 2];
            }
        }
}

// ---- atlas layout ------------------------------------------------------------------------------
static bool shelf_rect(const VecI& ws, const VecI& hs, i64 W, VecI& cx, VecI& cy) {
    i64 x = 0, y = 0, rowh = 0;
    for (size_t i = 0; i < ws.size(); i++) {
        i64 w = ws[i], h = hs[i];
        if (w > W || h > W) return false;
        if (x + w > W) {
            y += rowh;
            x = 0;
            rowh = 0;
        }
        if (y + h > W) return false;
        cx[i] = x;
        cy[i] = y;
        x += w;
        if (h > rowh) rowh = h;
    }
    return true;
}

static void footprints(double k, const VecD& base_w, const VecD& base_h, int pad, int gap, i64 smin, VecI& ws,
                       VecI& hs) {
    for (size_t i = 0; i < base_w.size(); i++) {  // every item is a pair chart
        i64 s = std::max(smin, (i64)(k * base_w[i]));
        ws[i] = s + gap + 2 * pad;
        hs[i] = ws[i];
    }
    (void)base_h;
}

static void assign_pair(const VecD& P, const VecI& F, i64 f, int slot, i64 s, i64 x0, i64 y0, int pad, int gap,
                        VecD& tc) {
    int best = 0;
    double bestc = 2.0;
    for (int k = 0; k < 3; k++) {
        i64 a = F[f * 3 + k], b = F[f * 3 + (k + 1) % 3], c = F[f * 3 + (k + 2) % 3];
        double ux = P[b * 3] - P[a * 3], uy = P[b * 3 + 1] - P[a * 3 + 1], uz = P[b * 3 + 2] - P[a * 3 + 2];
        double wx = P[c * 3] - P[a * 3], wy = P[c * 3 + 1] - P[a * 3 + 1], wz = P[c * 3 + 2] - P[a * 3 + 2];
        double den = std::sqrt((ux * ux + uy * uy + uz * uz) * (wx * wx + wy * wy + wz * wz));
        double cs = den > 0 ? (ux * wx + uy * wy + uz * wz) / den : 1.0;
        if (cs < bestc) {
            bestc = cs;
            best = k;
        }
    }
    x0 = x0 + pad;
    y0 = y0 + pad;
    int k0 = best, k1 = (best + 1) % 3, k2 = (best + 2) % 3;
    double* t = &tc[f * 6];
    if (slot == 0) {
        t[k0 * 2] = (double)x0;
        t[k0 * 2 + 1] = (double)y0;
        t[k1 * 2] = (double)(x0 + s);
        t[k1 * 2 + 1] = (double)y0;
        t[k2 * 2] = (double)x0;
        t[k2 * 2 + 1] = (double)(y0 + s);
    } else {
        t[k0 * 2] = (double)(x0 + gap + s);
        t[k0 * 2 + 1] = (double)(y0 + gap + s);
        t[k1 * 2] = (double)(x0 + gap);
        t[k1 * 2 + 1] = (double)(y0 + gap + s);
        t[k2 * 2] = (double)(x0 + gap + s);
        t[k2 * 2 + 1] = (double)(y0 + gap);
    }
}

struct Layout {
    VecD tc;          // (m, 3, 2)
    VecI cell;        // (m, 4)
    VecI item_start;  // CSR over charts
    VecI item_faces;
    i64 pair_charts = 0;
    double texels_per_unit = 0;
};

static bool layout_atlas(const VecD& S, const VecI& SF, i64 W, Layout& L) {
    i64 m = (i64)SF.size() / 3;
    VecD areas(m);
    for (i64 f = 0; f < m; f++) {
        const double* a = &S[SF[f * 3] * 3];
        const double* b = &S[SF[f * 3 + 1] * 3];
        const double* c = &S[SF[f * 3 + 2] * 3];
        double cx, cy, cz;
        np::cross(b[0] - a[0], b[1] - a[1], b[2] - a[2], c[0] - a[0], c[1] - a[1], c[2] - a[2], cx, cy, cz);
        areas[f] = np::norm3(cx, cy, cz) / 2;
    }
    VecD neg(m);
    for (i64 f = 0; f < m; f++) neg[f] = -areas[f];
    VecI curved = np::argsort_stable(neg);
    i64 n_pair = (m + 1) / 2;
    VecD base_w(n_pair);
    for (i64 i = 0; i < n_pair; i++) base_w[i] = std::sqrt(2 * np::fmax0(areas[curved[2 * i]], 1e-30));
    VecD base_h = base_w;
    VecD negh(n_pair);
    for (i64 i = 0; i < n_pair; i++) negh[i] = -base_h[i];
    VecI order = np::argsort_stable(negh);
    VecD bw(n_pair), bh(n_pair);
    for (i64 i = 0; i < n_pair; i++) {
        bw[i] = base_w[order[i]];
        bh[i] = base_h[order[i]];
    }
    VecI ws(n_pair), hs(n_pair), cx(n_pair), cy(n_pair);
    footprints(0.0, bw, bh, PAD, GAP, 2, ws, hs);
    if (!shelf_rect(ws, hs, W, cx, cy)) return false;
    double mw = 0, mh = 0;  // bw.max(initial=0)
    for (double v : bw) mw = np::fmax0(mw, v);
    for (double v : bh) mh = np::fmax0(mh, v);
    double mx = mw;  // Python max(a, b, c)
    if (mh > mx) mx = mh;
    if (1e-30 > mx) mx = 1e-30;
    double lo = 0.0, hi = (double)W / mx;
    for (int it = 0; it < 40; it++) {
        double k = 0.5 * (lo + hi);
        footprints(k, bw, bh, PAD, GAP, 2, ws, hs);
        if (shelf_rect(ws, hs, W, cx, cy)) lo = k;
        else hi = k;
    }
    double k = lo;
    footprints(k, bw, bh, PAD, GAP, 2, ws, hs);
    shelf_rect(ws, hs, W, cx, cy);

    L.tc.assign(m * 6, 0.0);
    L.cell.assign(m * 4, 0);
    VecI item_faces, item_start{0};
    for (i64 oi = 0; oi < n_pair; oi++) {
        i64 it = order[oi];
        i64 x0 = cx[oi], y0 = cy[oi], w = ws[oi], h = hs[oi];
        i64 s = w - GAP - 2 * PAD;
        VecI fl{curved[2 * it]};
        if (2 * it + 1 < m) fl.push_back(curved[2 * it + 1]);
        for (int slot = 0; slot < (int)fl.size(); slot++) {
            i64 f = fl[slot];
            assign_pair(S, SF, f, slot, s, x0, y0, PAD, GAP, L.tc);
            L.cell[f * 4] = x0;
            L.cell[f * 4 + 1] = y0;
            L.cell[f * 4 + 2] = w;
            L.cell[f * 4 + 3] = h;
        }
        item_faces.insert(item_faces.end(), fl.begin(), fl.end());
        item_start.push_back((i64)item_faces.size());
    }
    // shuffled chart order (bake.py: rng(0).permutation) - kept for parity of the CSR arrays
    i64 n_items = (i64)item_start.size() - 1;
    np::Generator rng(0);
    VecI perm = rng.permutation(n_items);
    L.item_start.assign(1, 0);
    L.item_faces.clear();
    for (i64 i : perm) {
        L.item_faces.insert(L.item_faces.end(), item_faces.begin() + item_start[i], item_faces.begin() + item_start[i + 1]);
        L.item_start.push_back((i64)L.item_faces.size());
    }
    L.pair_charts = n_pair;
    L.texels_per_unit = k;
    return true;
}

// ---- appearance sampling --------------------------------------------------------------------------
static inline double lin(double c) {
    double x = c / 255.0;
    return x <= 0.04045 ? x / 12.92 : np::py_pow((x + 0.055) / 1.055, 2.4);
}
static inline int to_srgb8(double x) {
    if (x <= 0.0) return 0;
    if (x >= 1.0) return 255;
    double y = x <= 0.0031308 ? x * 12.92 : 1.055 * np::py_pow(x, 1 / 2.4) - 0.055;
    return (int)(y * 255.0 + 0.5);
}

struct TexPack {
    std::vector<uint8_t> tex;
    VecI off, w, h, mat_tex, mat_mrtex;
};

static void tex_bilinear(const TexPack& T, i64 t, double u, double v, double* out) {
    i64 w = T.w[t], h = T.h[t], off = T.off[t];
    double x = (u - std::floor(u)) * (double)w - 0.5;
    double y = (1.0 - (v - std::floor(v))) * (double)h - 0.5;
    i64 x0 = (i64)std::floor(x), y0 = (i64)std::floor(y);
    double fx = x - (double)x0, fy = y - (double)y0;
    for (int c = 0; c < 4; c++) out[c] = 0.0;
    for (int dy = 0; dy < 2; dy++) {
        i64 yy = np::floor_mod(y0 + dy, h);
        double wy = dy == 1 ? fy : 1.0 - fy;
        for (int dx = 0; dx < 2; dx++) {
            i64 xx = np::floor_mod(x0 + dx, w);
            double wx = dx == 1 ? fx : 1.0 - fx;
            i64 base = off + (yy * w + xx) * 4;
            double ww = wx * wy;
            for (int c = 0; c < 4; c++) out[c] += ww * (double)T.tex[base + c];
        }
    }
}

struct BakeIn {
    const VecI* face_mat;
    const VecD* corner_uv;
    bool has_uv;
    const VecD* corner_col;
    bool has_col;
    VecD mat_color, mat_mr;  // (k, 4), (k, 2)
    TexPack tp;
};

static void sample_color(const BakeIn& B, i64 g, double b0, double b1, double b2, double* out, double* tmp) {
    i64 m = (*B.face_mat)[g];
    double r = B.mat_color[m * 4], gg = B.mat_color[m * 4 + 1], bb = B.mat_color[m * 4 + 2], a = B.mat_color[m * 4 + 3];
    i64 t = B.tp.mat_tex[m];
    if (t >= 0 && B.has_uv) {
        const double* uv = &(*B.corner_uv)[g * 6];
        double u = b0 * uv[0] + b1 * uv[2] + b2 * uv[4];
        double v = b0 * uv[1] + b1 * uv[3] + b2 * uv[5];
        tex_bilinear(B.tp, t, u, v, tmp);
        r *= lin(tmp[0]);
        gg *= lin(tmp[1]);
        bb *= lin(tmp[2]);
        a *= tmp[3] / 255.0;
    }
    if (B.has_col) {
        const double* cc = &(*B.corner_col)[g * 12];
        for (int c = 0; c < 4; c++) {
            double val = b0 * cc[c] + b1 * cc[4 + c] + b2 * cc[8 + c];
            if (c == 0) r *= val;
            else if (c == 1) gg *= val;
            else if (c == 2) bb *= val;
            else a *= val;
        }
    }
    out[0] = r;
    out[1] = gg;
    out[2] = bb;
    out[3] = a;
}

static void sample_mr(const BakeIn& B, i64 g, double b0, double b1, double b2, double* tmp, double& metal,
                      double& rough) {
    i64 m = (*B.face_mat)[g];
    metal = B.mat_mr[m * 2];
    rough = B.mat_mr[m * 2 + 1];
    i64 t = B.tp.mat_mrtex[m];
    if (t >= 0 && B.has_uv) {
        const double* uv = &(*B.corner_uv)[g * 6];
        double u = b0 * uv[0] + b1 * uv[2] + b2 * uv[4];
        double v = b0 * uv[1] + b1 * uv[3] + b2 * uv[5];
        tex_bilinear(B.tp, t, u, v, tmp);
        rough *= tmp[1] / 255.0;
        metal *= tmp[2] / 255.0;
    }
}

static inline double closest2d(double px, double py, double ax, double ay, double bx, double by, double cx, double cy,
                               double* out) {
    double v0x = bx - ax, v0y = by - ay, v1x = cx - ax, v1y = cy - ay, v2x = px - ax, v2y = py - ay;
    double den = v0x * v1y - v1x * v0y;
    if (den != 0.0) {
        double l1 = (v2x * v1y - v1x * v2y) / den;
        double l2 = (v0x * v2y - v2x * v0y) / den;
        double l0 = 1.0 - l1 - l2;
        if (l0 >= 0 && l1 >= 0 && l2 >= 0) {
            out[0] = l0;
            out[1] = l1;
            out[2] = l2;
            return 0.0;
        }
    }
    double best = INF;
    for (int e = 0; e < 3; e++) {
        double sx, sy, tx, ty;
        if (e == 0) {
            sx = ax; sy = ay; tx = bx; ty = by;
        } else if (e == 1) {
            sx = bx; sy = by; tx = cx; ty = cy;
        } else {
            sx = cx; sy = cy; tx = ax; ty = ay;
        }
        double dx = tx - sx, dy = ty - sy;
        double ll = dx * dx + dy * dy;
        double t = ll > 0 ? ((px - sx) * dx + (py - sy) * dy) / ll : 0.0;
        t = std::min(1.0, std::max(0.0, t));
        double qx = sx + t * dx, qy = sy + t * dy;
        double d = std::sqrt((px - qx) * (px - qx) + (py - qy) * (py - qy));
        if (d < best) {
            best = d;
            if (e == 0) {
                out[0] = 1 - t; out[1] = t; out[2] = 0.0;
            } else if (e == 1) {
                out[0] = 0.0; out[1] = 1 - t; out[2] = t;
            } else {
                out[0] = t; out[1] = 0.0; out[2] = 1 - t;
            }
        }
    }
    return best;
}

static inline uint8_t clamp8(int v) { return (uint8_t)std::min(255, std::max(0, v)); }

int choose_atlas_size(i64 n_faces, int requested) {
    if (requested) return requested;
    if (n_faces <= 4000) return 1024;
    if (n_faces <= 20000) return 2048;
    return 4096;
}

BakeResult bake(const VecD& S, const VecI& SF, const PreparedMesh& prep, const Asset& asset, const VecI& vertex_map,
                int atlas_size, bool bake_color, bool bake_normal, bool bake_mr, double crease_angle,
                const BVH* orig_bvh) {
    i64 m = (i64)SF.size() / 3;
    double crease_cos = fm::cos(crease_angle * (PI / 180.0));
    VecD snrm, sfn;
    corner_normals(S, SF, crease_cos, snrm, sfn);
    i64 W = choose_atlas_size(m, atlas_size);
    Layout L;
    bool ok;
    while (true) {
        ok = layout_atlas(S, SF, W, L);
        if (ok || W >= 8192) break;
        W *= 2;
    }
    if (!ok) throw std::runtime_error("too many faces for an 8192 texture atlas");

    const VecD& OP = prep.positions;
    const VecI& OF = prep.faces;
    BVH own;
    if (!orig_bvh) {
        own = BVH(OP, OF);
        orig_bvh = &own;
    }
    BVHView T = view(*orig_bvh);
    i64 nOF = (i64)OF.size() / 3;
    VecI reps(nOF * 3);
    for (i64 i = 0; i < nOF * 3; i++) reps[i] = vertex_map[OF[i]];
    VecD onrm, ofn;
    corner_normals(OP, OF, fm::cos(crease_angle * (PI / 180.0)), onrm, ofn);

    BakeIn B;
    const VecI& fo = prep.face_orig;
    VecI face_mat(fo.size());
    for (size_t i = 0; i < fo.size(); i++) face_mat[i] = asset.face_material[fo[i]];
    B.face_mat = &face_mat;
    B.has_uv = asset.has_uv;
    VecD corner_uv, corner_col;
    if (B.has_uv) {
        corner_uv.resize(fo.size() * 6);
        for (size_t i = 0; i < fo.size(); i++)
            for (int k = 0; k < 6; k++) corner_uv[i * 6 + k] = asset.corner_uv[fo[i] * 6 + k];
    }
    B.corner_uv = &corner_uv;
    B.has_col = asset.has_col;
    if (B.has_col) {
        corner_col.resize(fo.size() * 12);
        for (size_t i = 0; i < fo.size(); i++)
            for (int k = 0; k < 12; k++) corner_col[i * 12 + k] = asset.corner_color[fo[i] * 12 + k];
    }
    B.corner_col = &corner_col;
    for (auto& mt : asset.materials) {
        for (int k = 0; k < 4; k++) B.mat_color.push_back(mt.color[k]);
        B.mat_mr.push_back(mt.metallic);
        B.mat_mr.push_back(mt.roughness);
    }
    // _pack_textures
    std::vector<const Image*> imgs;
    for (auto& mt : asset.materials) {
        for (int which = 0; which < 2; which++) {
            const ImagePtr& im = which == 0 ? mt.texture : mt.mr_texture;
            VecI& lst = which == 0 ? B.tp.mat_tex : B.tp.mat_mrtex;
            if (!im) lst.push_back(-1);
            else {
                lst.push_back((i64)imgs.size());
                imgs.push_back(im.get());
            }
        }
    }
    Image dummy;
    dummy.w = dummy.h = 1;
    dummy.px.assign(4, 0);
    if (imgs.empty()) imgs.push_back(&dummy);
    i64 o = 0;
    for (auto* im : imgs) {
        B.tp.off.push_back(o);
        B.tp.w.push_back(im->w);
        B.tp.h.push_back(im->h);
        B.tp.tex.insert(B.tp.tex.end(), im->px.begin(), im->px.end());
        o += (i64)im->px.size();
    }
    // "large" simplified triangle: longest edge above 8x the original mean edge length
    double oe = 1.0;
    if (nOF) {
        VecD el(nOF);
        for (i64 f = 0; f < nOF; f++) {
            const double* a = &OP[OF[f * 3] * 3];
            const double* b = &OP[OF[f * 3 + 1] * 3];
            el[f] = np::norm3(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
        }
        oe = np::mean(el);
    }
    double big_face2 = np::py_pow(8.0 * oe, 2.0);
    bool do_color = bake_color && asset.has_appearance();
    bool do_mr = bake_mr && asset.has_mr_variation();
    bool do_normal = bake_normal;
    double dilate = (double)PAD + 0.75;

    std::vector<uint8_t> color(W * W * 4, 0), normal(W * W * 3, 0), mr(W * W * 3, 0), covered(W * W, 0);
    std::vector<float> owner(W * W, std::numeric_limits<float>::infinity());
    i64 n_items = (i64)L.item_start.size() - 1;
    const VecD& tc = L.tc;
    const VecI& cell = L.cell;
    parallel_for(n_items, [&](i64 is, i64 ie) {
        i64 stack[256];
        double sd[256], o6[6], t6[6], bc[3], col[4], tmp[4], keep6[6];
        for (i64 it = is; it < ie; it++) {
            for (i64 jj = L.item_start[it]; jj < L.item_start[it + 1]; jj++) {
                i64 f = L.item_faces[jj];
                i64 A = SF[f * 3], Bv = SF[f * 3 + 1], Cc = SF[f * 3 + 2];
                double ax = tc[f * 6], ay = tc[f * 6 + 1];
                double bx = tc[f * 6 + 2], by = tc[f * 6 + 3];
                double cx = tc[f * 6 + 4], cy = tc[f * 6 + 5];
                i64 x0 = cell[f * 4], y0 = cell[f * 4 + 1], cw = cell[f * 4 + 2], ch = cell[f * 4 + 3];
                double nfx = sfn[f * 3], nfy = sfn[f * 3 + 1], nfz = sfn[f * 3 + 2];
                double dm = 0.0;
                for (int e0 = 0; e0 < 3; e0++) {
                    int e1_ = (e0 + 1) % 3;
                    const double* p0 = &S[SF[f * 3 + e0] * 3];
                    const double* p1 = &S[SF[f * 3 + e1_] * 3];
                    double dd = (p0[0] - p1[0]) * (p0[0] - p1[0]) + (p0[1] - p1[1]) * (p0[1] - p1[1]) +
                                (p0[2] - p1[2]) * (p0[2] - p1[2]);
                    if (dd > dm) dm = dd;
                }
                double rad2f = 4.0 * dm + 1e-12;
                i64 hint0 = -1;
                const double* SA = &S[A * 3];
                const double* SB = &S[Bv * 3];
                const double* SC = &S[Cc * 3];
                double e1x = SB[0] - SA[0], e1y = SB[1] - SA[1], e1z = SB[2] - SA[2];
                double e2x = SC[0] - SA[0], e2y = SC[1] - SA[1], e2z = SC[2] - SA[2];
                double du1 = bx - ax, dv1 = -(by - ay), du2 = cx - ax, dv2 = -(cy - ay);
                double det = du1 * dv2 - du2 * dv1;
                double r = det != 0 ? 1.0 / det : 0.0;
                double Tx = (e1x * dv2 - e2x * dv1) * r, Ty = (e1y * dv2 - e2y * dv1) * r, Tz = (e1z * dv2 - e2z * dv1) * r;
                double Bx = (e2x * du1 - e1x * du2) * r, By = (e2y * du1 - e1y * du2) * r, Bz = (e2z * du1 - e1z * du2) * r;
                double hx = nfy * Tz - nfz * Ty, hy = nfz * Tx - nfx * Tz, hz = nfx * Ty - nfy * Tx;
                double hand = hx * Bx + hy * By + hz * Bz >= 0 ? 1.0 : -1.0;
                i64 xs = std::max(x0, (i64)std::floor(std::min(std::min(ax, bx), cx) - dilate));
                i64 xe = std::min(std::min(x0 + cw, (i64)std::ceil(std::max(std::max(ax, bx), cx) + dilate) + 1), W);
                i64 ys = std::max(y0, (i64)std::floor(std::min(std::min(ay, by), cy) - dilate));
                i64 ye = std::min(std::min(y0 + ch, (i64)std::ceil(std::max(std::max(ay, by), cy) + dilate) + 1), W);
                i64 hint = -1;
                for (i64 ty = ys; ty < ye; ty++) {
                    double ya = (double)ty - dilate, yb = (double)ty + 1 + dilate;
                    double rx0 = 1e30, rx1 = -1e30;
                    for (int ed = 0; ed < 3; ed++) {
                        double x1, y1, x2, y2;
                        if (ed == 0) {
                            x1 = ax; y1 = ay; x2 = bx; y2 = by;
                        } else if (ed == 1) {
                            x1 = bx; y1 = by; x2 = cx; y2 = cy;
                        } else {
                            x1 = cx; y1 = cy; x2 = ax; y2 = ay;
                        }
                        double lo_y = std::min(y1, y2), hi_y = std::max(y1, y2);
                        if (hi_y < ya || lo_y > yb) continue;
                        if (y2 == y1) {
                            rx0 = std::min(std::min(rx0, x1), x2);
                            rx1 = std::max(std::max(rx1, x1), x2);
                            continue;
                        }
                        double yys[2] = {std::max(ya, lo_y), std::min(yb, hi_y)};
                        for (double yy : yys) {
                            double xx = x1 + (x2 - x1) * (yy - y1) / (y2 - y1);
                            rx0 = std::min(rx0, xx);
                            rx1 = std::max(rx1, xx);
                        }
                    }
                    if (rx1 < rx0) continue;
                    i64 sx = std::max(xs, (i64)std::floor(rx0 - dilate));
                    i64 ex = std::min(xe, (i64)std::ceil(rx1 + dilate) + 1);
                    for (i64 tx = sx; tx < ex; tx++) {
                        double rad2 = rad2f;
                        double d = closest2d((double)tx + 0.5, (double)ty + 0.5, ax, ay, bx, by, cx, cy, bc);
                        if (d > dilate || d >= (double)owner[ty * W + tx]) continue;
                        double b0 = bc[0], b1 = bc[1], b2 = bc[2];
                        double px = b0 * SA[0] + b1 * SB[0] + b2 * SC[0];
                        double py = b0 * SA[1] + b1 * SB[1] + b2 * SC[1];
                        double pz = b0 * SA[2] + b1 * SB[2] + b2 * SC[2];
                        i64 g0 = -1;
                        double d0 = 0.0;
                        if (dm > big_face2)
                            g0 = bvh_closest(T, px, py, pz, hint0, 0, reps.data(), A, Bv, Cc, ofn.data(), nfx, nfy, nfz,
                                             stack, sd, 256, o6, t6, INF, d0);
                        if (g0 >= 0) {
                            hint0 = g0;
                            for (int k6 = 0; k6 < 6; k6++) keep6[k6] = o6[k6];
                            double rr = 3.0 * std::sqrt(d0) + 0.1 * std::sqrt(dm);
                            rad2 = std::min(rad2, rr * rr + 1e-20);
                        }
                        double d2tmp;
                        i64 g = bvh_closest(T, px, py, pz, hint, 2, reps.data(), A, Bv, Cc, ofn.data(), nfx, nfy, nfz,
                                            stack, sd, 256, o6, t6, rad2, d2tmp);
                        if (g < 0)
                            g = bvh_closest(T, px, py, pz, -1, 1, reps.data(), A, Bv, Cc, ofn.data(), nfx, nfy, nfz,
                                            stack, sd, 256, o6, t6, rad2, d2tmp);
                        if (g < 0 && g0 >= 0) {
                            g = g0;
                            for (int k6 = 0; k6 < 6; k6++) o6[k6] = keep6[k6];
                        }
                        if (g < 0)
                            g = bvh_closest(T, px, py, pz, -1, 0, reps.data(), A, Bv, Cc, ofn.data(), nfx, nfy, nfz,
                                            stack, sd, 256, o6, t6, INF, d2tmp);
                        if (g < 0) continue;
                        hint = g;
                        double c0 = o6[0], c1 = o6[1], c2 = o6[2];
                        owner[ty * W + tx] = (float)d;
                        covered[ty * W + tx] = 1;
                        if (do_color) {
                            sample_color(B, g, c0, c1, c2, col, tmp);
                            uint8_t* cp = &color[(ty * W + tx) * 4];
                            cp[0] = (uint8_t)to_srgb8(col[0]);
                            cp[1] = (uint8_t)to_srgb8(col[1]);
                            cp[2] = (uint8_t)to_srgb8(col[2]);
                            cp[3] = clamp8((int)(col[3] * 255.0 + 0.5));
                        }
                        if (do_mr) {
                            double metal, rough;
                            sample_mr(B, g, c0, c1, c2, tmp, metal, rough);
                            uint8_t* mp = &mr[(ty * W + tx) * 3];
                            mp[0] = 255;
                            mp[1] = clamp8((int)(rough * 255.0 + 0.5));
                            mp[2] = clamp8((int)(metal * 255.0 + 0.5));
                        }
                        if (do_normal) {
                            const double* on = &onrm[g * 9];
                            double onx = c0 * on[0] + c1 * on[3] + c2 * on[6];
                            double ony = c0 * on[1] + c1 * on[4] + c2 * on[7];
                            double onz = c0 * on[2] + c1 * on[5] + c2 * on[8];
                            const double* sn = &snrm[f * 9];
                            double Nx = b0 * sn[0] + b1 * sn[3] + b2 * sn[6];
                            double Ny = b0 * sn[1] + b1 * sn[4] + b2 * sn[7];
                            double Nz = b0 * sn[2] + b1 * sn[5] + b2 * sn[8];
                            double nl = std::sqrt(Nx * Nx + Ny * Ny + Nz * Nz);
                            double ol = std::sqrt(onx * onx + ony * ony + onz * onz);
                            uint8_t* np_ = &normal[(ty * W + tx) * 3];
                            if (nl > 0 && ol > 0) {
                                Nx /= nl;
                                Ny /= nl;
                                Nz /= nl;
                                onx /= ol;
                                ony /= ol;
                                onz /= ol;
                                double dt = Tx * Nx + Ty * Ny + Tz * Nz;
                                double tx_ = Tx - dt * Nx, ty_ = Ty - dt * Ny, tz_ = Tz - dt * Nz;
                                double tl = std::sqrt(tx_ * tx_ + ty_ * ty_ + tz_ * tz_);
                                if (tl > 0) {
                                    tx_ /= tl;
                                    ty_ /= tl;
                                    tz_ /= tl;
                                }
                                double bx_ = hand * (Ny * tz_ - Nz * ty_);
                                double by_ = hand * (Nz * tx_ - Nx * tz_);
                                double bz_ = hand * (Nx * ty_ - Ny * tx_);
                                double vx = onx * tx_ + ony * ty_ + onz * tz_;
                                double vy = onx * bx_ + ony * by_ + onz * bz_;
                                double vz = onx * Nx + ony * Ny + onz * Nz;
                                if (vz < 0.05) {
                                    vz = 0.05;
                                    double s2 = vx * vx + vy * vy;
                                    if (s2 > 0) {
                                        double kk = std::sqrt((1 - vz * vz) / s2);
                                        vx *= kk;
                                        vy *= kk;
                                    }
                                }
                                double vl = std::sqrt(vx * vx + vy * vy + vz * vz);
                                np_[0] = clamp8((int)((vx / vl * 0.5 + 0.5) * 255 + 0.5));
                                np_[1] = clamp8((int)((vy / vl * 0.5 + 0.5) * 255 + 0.5));
                                np_[2] = clamp8((int)((vz / vl * 0.5 + 0.5) * 255 + 0.5));
                            } else {
                                np_[0] = 128;
                                np_[1] = 128;
                                np_[2] = 255;
                            }
                        }
                    }
                }
            }
        }
    }, 64);

    // fill the unused atlas area (exact integer mean, truncated to uint8)
    i64 ncov = 0;
    for (auto c : covered) ncov += c;
    if (do_color && ncov) {
        uint64_t s[4] = {0, 0, 0, 0};
        for (i64 i = 0; i < W * W; i++)
            if (covered[i])
                for (int c = 0; c < 4; c++) s[c] += color[i * 4 + c];
        uint8_t mean[4];
        for (int c = 0; c < 4; c++) mean[c] = (uint8_t)((double)s[c] / (double)ncov);
        for (i64 i = 0; i < W * W; i++)
            if (!covered[i])
                for (int c = 0; c < 4; c++) color[i * 4 + c] = mean[c];
    }
    if (do_normal)
        for (i64 i = 0; i < W * W; i++)
            if (!covered[i]) {
                normal[i * 3] = 128;
                normal[i * 3 + 1] = 128;
                normal[i * 3 + 2] = 255;
            }
    if (do_mr && ncov) {
        uint64_t s[3] = {0, 0, 0};
        for (i64 i = 0; i < W * W; i++)
            if (covered[i])
                for (int c = 0; c < 3; c++) s[c] += mr[i * 3 + c];
        uint8_t mean[3];
        for (int c = 0; c < 3; c++) mean[c] = (uint8_t)((double)s[c] / (double)ncov);
        for (i64 i = 0; i < W * W; i++)
            if (!covered[i])
                for (int c = 0; c < 3; c++) mr[i * 3 + c] = mean[c];
    }

    BakeResult R;
    R.positions.resize(m * 9);
    for (i64 f = 0; f < m; f++)
        for (int k = 0; k < 3; k++)
            for (int c = 0; c < 3; c++) R.positions[(f * 3 + k) * 3 + c] = S[SF[f * 3 + k] * 3 + c];
    R.normals = snrm;
    R.uv.resize(m * 6);
    for (i64 i = 0; i < m * 3; i++) {
        R.uv[i * 2] = tc[i * 2] / (double)W;
        R.uv[i * 2 + 1] = 1.0 - tc[i * 2 + 1] / (double)W;
    }
    auto mk = [&](std::vector<uint8_t>& px, int ch) {
        ImagePtr im = std::make_shared<Image>();
        im->w = im->h = (int)W;
        im->channels = ch;
        im->px = std::move(px);
        return im;
    };
    if (do_color) R.color = mk(color, 4);
    if (do_normal) R.normal_map = mk(normal, 3);
    if (do_mr) R.mr = mk(mr, 3);
    R.atlas_size = (int)W;
    R.pair_charts = L.pair_charts;
    R.texels_per_unit = L.texels_per_unit;
    if (!asset.materials.empty()) {
        const Material& b = asset.materials[0];
        for (int k = 0; k < 4; k++) R.base_color[k] = b.color[k];
        R.metallic = b.metallic;
        R.roughness = b.roughness;
    }
    return R;
}

}  // namespace faqem
