"""Stage 2 - appearance transfer (paper Sec. 3.2.3).

The simplified mesh gets a fresh per-triangle texture atlas ("per-triangle flattening",
independent of the original UV layout, so no seam / bleeding constraints on the
simplification). Every texel is a point p on the simplified surface; its correspondence
T(p) on the original mesh follows the collapse lineage recorded during simplification
(successive mapping): only original triangles whose vertices collapsed into the corners
of p's triangle are admissible, and among those the closest point is taken (with a
same-side normal test that stops thin walls from grabbing the colour of their back face).
This keeps the paper's locality guarantee while being exact inside each lineage region.

Baked maps:  base colour (textures x factors x vertex colours, sRGB)
             normal map (tangent space, glTF / MikkTSpace convention) - recovers the fine
             surface detail of the dense mesh on the low-poly geometry (extension)
             metallic-roughness (when the source has spatially varying PBR values)
"""
import numpy as np
from numba import njit, prange

from .bvh import BVH, bvh_closest

PAD = 2          # gutter texels around every chart
GAP = 2          # separation between the two triangles sharing a cell


# ------------------------------------------------------------------------------------
# normals
# ------------------------------------------------------------------------------------
@njit(cache=True)
def corner_normals(P, F, crease_cos):
    """Crease-aware per-corner normals: average area-weighted normals of the faces around
    the vertex whose normal is within the crease angle of the corner's face."""
    nV = P.shape[0]; nF = F.shape[0]
    fn = np.zeros((nF, 3))
    fu = np.zeros((nF, 3))
    for f in range(nF):
        a = F[f, 0]; b = F[f, 1]; c = F[f, 2]
        ux = P[b, 0] - P[a, 0]; uy = P[b, 1] - P[a, 1]; uz = P[b, 2] - P[a, 2]
        wx = P[c, 0] - P[a, 0]; wy = P[c, 1] - P[a, 1]; wz = P[c, 2] - P[a, 2]
        nx = uy * wz - uz * wy; ny = uz * wx - ux * wz; nz = ux * wy - uy * wx
        fn[f, 0] = nx; fn[f, 1] = ny; fn[f, 2] = nz
        l = np.sqrt(nx * nx + ny * ny + nz * nz)
        if l > 0:
            fu[f, 0] = nx / l; fu[f, 1] = ny / l; fu[f, 2] = nz / l
    deg = np.zeros(nV + 1, np.int64)
    for f in range(nF):
        for k in range(3):
            deg[F[f, k] + 1] += 1
    for i in range(nV):
        deg[i + 1] += deg[i]
    fill = deg[:nV].copy()
    vf = np.empty(3 * nF, np.int64)
    for f in range(nF):
        for k in range(3):
            v = F[f, k]
            vf[fill[v]] = f
            fill[v] += 1
    out = np.zeros((nF, 3, 3))
    for f in range(nF):
        for k in range(3):
            v = F[f, k]
            sx = 0.0; sy = 0.0; sz = 0.0
            for j in range(deg[v], deg[v + 1]):
                g = vf[j]
                if fu[g, 0] * fu[f, 0] + fu[g, 1] * fu[f, 1] + fu[g, 2] * fu[f, 2] >= crease_cos:
                    sx += fn[g, 0]; sy += fn[g, 1]; sz += fn[g, 2]
            l = np.sqrt(sx * sx + sy * sy + sz * sz)
            if l > 0:
                out[f, k, 0] = sx / l; out[f, k, 1] = sy / l; out[f, k, 2] = sz / l
            else:
                out[f, k, 0] = fu[f, 0]; out[f, k, 1] = fu[f, 1]; out[f, k, 2] = fu[f, 2]
    return out, fu


# ------------------------------------------------------------------------------------
# atlas layout: two triangles per square cell (size ~ sqrt(area)), shelf-packed at one
#               binary-searched texel density
# ------------------------------------------------------------------------------------
@njit(cache=True)
def _shelf_rect(ws, hs, W, cx, cy):
    x = 0; y = 0; rowh = 0
    for i in range(ws.shape[0]):
        w = ws[i]; h = hs[i]
        if w > W or h > W:
            return False
        if x + w > W:
            y += rowh; x = 0; rowh = 0
        if y + h > W:
            return False
        cx[i] = x; cy[i] = y
        x += w
        if h > rowh:
            rowh = h
    return True


@njit(cache=True)
def _footprints(k, base_w, base_h, is_pair, pad, gap, smin, ws, hs):
    for i in range(base_w.shape[0]):
        if is_pair[i]:
            s = max(smin, int(k * base_w[i]))
            ws[i] = s + gap + 2 * pad
            hs[i] = ws[i]
        else:
            ws[i] = max(smin, int(np.ceil(k * base_w[i]))) + 2 * pad + 1
            hs[i] = max(smin, int(np.ceil(k * base_h[i]))) + 2 * pad + 1


@njit(cache=True)
def _assign_pair(P, F, f, slot, s, x0, y0, pad, gap, tc):
    """Right-triangle placement inside a square cell (largest angle at the right angle)."""
    best = 0; bestc = 2.0
    for k in range(3):
        a = F[f, k]; b = F[f, (k + 1) % 3]; c = F[f, (k + 2) % 3]
        ux = P[b, 0] - P[a, 0]; uy = P[b, 1] - P[a, 1]; uz = P[b, 2] - P[a, 2]
        wx = P[c, 0] - P[a, 0]; wy = P[c, 1] - P[a, 1]; wz = P[c, 2] - P[a, 2]
        den = np.sqrt((ux * ux + uy * uy + uz * uz) * (wx * wx + wy * wy + wz * wz))
        cs = (ux * wx + uy * wy + uz * wz) / den if den > 0 else 1.0
        if cs < bestc:
            bestc = cs; best = k
    x0 = x0 + pad; y0 = y0 + pad
    k0 = best; k1 = (best + 1) % 3; k2 = (best + 2) % 3
    if slot == 0:
        tc[f, k0, 0] = x0;         tc[f, k0, 1] = y0
        tc[f, k1, 0] = x0 + s;     tc[f, k1, 1] = y0
        tc[f, k2, 0] = x0;         tc[f, k2, 1] = y0 + s
    else:
        tc[f, k0, 0] = x0 + gap + s; tc[f, k0, 1] = y0 + gap + s
        tc[f, k1, 0] = x0 + gap;     tc[f, k1, 1] = y0 + gap + s
        tc[f, k2, 0] = x0 + gap + s; tc[f, k2, 1] = y0 + gap


def layout_atlas(S, SF, W):
    """Returns tc (m,3,2) texel coords, cell (m,4) chart rects, item CSR, or None if it doesn't fit."""
    m = len(SF)
    areas = np.linalg.norm(np.cross(S[SF[:, 1]] - S[SF[:, 0]], S[SF[:, 2]] - S[SF[:, 0]]), axis=1) / 2
    curved = np.argsort(-areas, kind="stable")
    n_pair = (len(curved) + 1) // 2
    base_w = np.sqrt(2 * np.maximum(areas[curved[0::2]], 1e-30)) if n_pair else np.zeros(0)
    base_h = base_w.copy()
    is_pair = np.ones(n_pair, bool)
    order = np.argsort(-base_h, kind="stable")
    bw, bh, ip = base_w[order], base_h[order], is_pair[order]
    ws = np.zeros(len(order), np.int64); hs = np.zeros(len(order), np.int64)
    cx = np.zeros(len(order), np.int64); cy = np.zeros(len(order), np.int64)
    _footprints(0.0, bw, bh, ip, PAD, GAP, 2, ws, hs)
    if not _shelf_rect(ws, hs, W, cx, cy):
        return None
    lo, hi = 0.0, W / max(float(bw.max(initial=0)), float(bh.max(initial=0)), 1e-30)
    for _ in range(40):
        k = 0.5 * (lo + hi)
        _footprints(k, bw, bh, ip, PAD, GAP, 2, ws, hs)
        if _shelf_rect(ws, hs, W, cx, cy):
            lo = k
        else:
            hi = k
    k = lo
    _footprints(k, bw, bh, ip, PAD, GAP, 2, ws, hs)
    _shelf_rect(ws, hs, W, cx, cy)

    tc = np.zeros((m, 3, 2))
    cell = np.zeros((m, 4), np.int64)
    item_faces, item_start = [], [0]
    for oi, it in enumerate(order):
        x0, y0, w, h = int(cx[oi]), int(cy[oi]), int(ws[oi]), int(hs[oi])
        s = w - GAP - 2 * PAD
        fl = [curved[2 * it]] + ([curved[2 * it + 1]] if 2 * it + 1 < len(curved) else [])
        for slot, f in enumerate(fl):
            _assign_pair(S, SF, int(f), slot, s, x0, y0, PAD, GAP, tc)
            cell[f] = (x0, y0, w, h)
        item_faces += fl
        item_start.append(len(item_faces))
    # shuffle the charts: prange hands each thread a contiguous block, and the charts above are
    # sorted largest-first, which would give one thread all the heavy work
    item_start = np.array(item_start, np.int64)
    item_faces = np.array(item_faces, np.int64)
    n_items = len(item_start) - 1
    perm = np.random.default_rng(0).permutation(n_items)
    lens = np.diff(item_start)[perm]
    new_start = np.r_[0, np.cumsum(lens)].astype(np.int64)
    new_faces = np.concatenate([item_faces[item_start[i]:item_start[i + 1]] for i in perm]) if n_items else item_faces
    return (tc, cell, new_start, new_faces.astype(np.int64),
            dict(pair_charts=n_pair, texels_per_unit=k))


# ------------------------------------------------------------------------------------
# appearance sampling on the original mesh
# ------------------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _lin(c):
    x = c / 255.0
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


@njit(cache=True, inline="always")
def _to_srgb8(x):
    if x <= 0.0:
        return 0
    if x >= 1.0:
        return 255
    y = x * 12.92 if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055
    return int(y * 255.0 + 0.5)


@njit(cache=True)
def _tex_bilinear(tex, off, w, h, u, v, out):
    """Bilinear RGBA sample (0..255 floats), repeat wrapping, OpenGL v-up uv."""
    x = (u - np.floor(u)) * w - 0.5
    y = (1.0 - (v - np.floor(v))) * h - 0.5
    x0 = int(np.floor(x)); y0 = int(np.floor(y))
    fx = x - x0; fy = y - y0
    for c in range(4):
        out[c] = 0.0
    for dy in range(2):
        yy = (y0 + dy) % h
        wy = fy if dy == 1 else 1.0 - fy
        for dx in range(2):
            xx = (x0 + dx) % w
            wx = fx if dx == 1 else 1.0 - fx
            base = off + (yy * w + xx) * 4
            ww = wx * wy
            for c in range(4):
                out[c] += ww * tex[base + c]


@njit(cache=True)
def _sample_color(g, b0, b1, b2, face_mat, corner_uv, has_uv, corner_col, has_col,
                  mat_color, mat_tex, tex, tex_off, tex_w, tex_h, out, tmp):
    """Linear RGBA of the original surface at face g, barycentric (b0, b1, b2)."""
    m = face_mat[g]
    r = mat_color[m, 0]; gg = mat_color[m, 1]; bb = mat_color[m, 2]; a = mat_color[m, 3]
    t = mat_tex[m]
    if t >= 0 and has_uv:
        u = b0 * corner_uv[g, 0, 0] + b1 * corner_uv[g, 1, 0] + b2 * corner_uv[g, 2, 0]
        v = b0 * corner_uv[g, 0, 1] + b1 * corner_uv[g, 1, 1] + b2 * corner_uv[g, 2, 1]
        _tex_bilinear(tex, tex_off[t], tex_w[t], tex_h[t], u, v, tmp)
        r *= _lin(tmp[0]); gg *= _lin(tmp[1]); bb *= _lin(tmp[2]); a *= tmp[3] / 255.0
    if has_col:
        for c in range(4):
            val = b0 * corner_col[g, 0, c] + b1 * corner_col[g, 1, c] + b2 * corner_col[g, 2, c]
            if c == 0:
                r *= val
            elif c == 1:
                gg *= val
            elif c == 2:
                bb *= val
            else:
                a *= val
    out[0] = r; out[1] = gg; out[2] = bb; out[3] = a


@njit(cache=True)
def _sample_mr(g, b0, b1, b2, face_mat, corner_uv, has_uv, mat_mr, mat_mrtex,
               tex, tex_off, tex_w, tex_h, tmp):
    m = face_mat[g]
    metal = mat_mr[m, 0]; rough = mat_mr[m, 1]
    t = mat_mrtex[m]
    if t >= 0 and has_uv:
        u = b0 * corner_uv[g, 0, 0] + b1 * corner_uv[g, 1, 0] + b2 * corner_uv[g, 2, 0]
        v = b0 * corner_uv[g, 0, 1] + b1 * corner_uv[g, 1, 1] + b2 * corner_uv[g, 2, 1]
        _tex_bilinear(tex, tex_off[t], tex_w[t], tex_h[t], u, v, tmp)
        rough *= tmp[1] / 255.0
        metal *= tmp[2] / 255.0
    return metal, rough


# ------------------------------------------------------------------------------------
# the bake kernel
# ------------------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _closest2d(px, py, ax, ay, bx, by, cx, cy, out):
    """Closest point of triangle abc to p in 2D: barycentric -> out[0:3], returns distance."""
    v0x = bx - ax; v0y = by - ay; v1x = cx - ax; v1y = cy - ay; v2x = px - ax; v2y = py - ay
    den = v0x * v1y - v1x * v0y
    if den != 0.0:
        l1 = (v2x * v1y - v1x * v2y) / den
        l2 = (v0x * v2y - v2x * v0y) / den
        l0 = 1.0 - l1 - l2
        if l0 >= 0 and l1 >= 0 and l2 >= 0:
            out[0] = l0; out[1] = l1; out[2] = l2
            return 0.0
    best = np.inf
    for e in range(3):
        if e == 0:
            sx = ax; sy = ay; tx = bx; ty = by
        elif e == 1:
            sx = bx; sy = by; tx = cx; ty = cy
        else:
            sx = cx; sy = cy; tx = ax; ty = ay
        dx = tx - sx; dy = ty - sy
        ll = dx * dx + dy * dy
        t = ((px - sx) * dx + (py - sy) * dy) / ll if ll > 0 else 0.0
        t = min(1.0, max(0.0, t))
        qx = sx + t * dx; qy = sy + t * dy
        d = np.sqrt((px - qx) ** 2 + (py - qy) ** 2)
        if d < best:
            best = d
            if e == 0:
                out[0] = 1 - t; out[1] = t; out[2] = 0.0
            elif e == 1:
                out[0] = 0.0; out[1] = 1 - t; out[2] = t
            else:
                out[0] = t; out[1] = 0.0; out[2] = 1 - t
    return best


@njit(cache=True, parallel=True)
def bake_kernel(W, S, SF, snrm, sfn, tc, cell, item_start, item_faces,
                P, F, idx, box, left, start, count, reps, ofn, onrm,
                face_mat, corner_uv, has_uv, corner_col, has_col, mat_color, mat_tex,
                mat_mr, mat_mrtex, tex, tex_off, tex_w, tex_h,
                do_color, do_normal, do_mr, dilate, big_face2):
    """Charts ("items", triangle pairs) are baked in parallel. cell[f] = (x0, y0, w, h) of the
    face's chart."""
    color = np.zeros((W, W, 4), np.uint8)
    normal = np.zeros((W, W, 3), np.uint8)
    mr = np.zeros((W, W, 3), np.uint8)
    owner = np.full((W, W), np.inf, np.float32)
    covered = np.zeros((W, W), np.uint8)
    n_items = item_start.shape[0] - 1
    for it in prange(n_items):
        stack = np.empty(256, np.int64)
        sd = np.empty(256)
        o6 = np.empty(6)
        t6 = np.empty(6)
        bc = np.empty(3)
        col = np.empty(4)
        tmp = np.empty(4)
        keep6 = np.empty(6)
        for jj in range(item_start[it], item_start[it + 1]):
            f = item_faces[jj]
            A = SF[f, 0]; B = SF[f, 1]; Cc = SF[f, 2]
            ax = tc[f, 0, 0]; ay = tc[f, 0, 1]
            bx = tc[f, 1, 0]; by = tc[f, 1, 1]
            cx = tc[f, 2, 0]; cy = tc[f, 2, 1]
            x0 = cell[f, 0]; y0 = cell[f, 1]; cw = cell[f, 2]; ch = cell[f, 3]
            nfx = sfn[f, 0]; nfy = sfn[f, 1]; nfz = sfn[f, 2]
            # search radius for the filtered correspondences: a few face diameters
            dm = 0.0
            for e0 in range(3):
                e1_ = (e0 + 1) % 3
                dd = (S[SF[f, e0], 0] - S[SF[f, e1_], 0]) ** 2 + (S[SF[f, e0], 1] - S[SF[f, e1_], 1]) ** 2 + \
                     (S[SF[f, e0], 2] - S[SF[f, e1_], 2]) ** 2
                if dd > dm:
                    dm = dd
            rad2f = 4.0 * dm + 1e-12
            hint0 = -1
            # tangent frame from the chart (u right, v up == -y)
            e1x = S[B, 0] - S[A, 0]; e1y = S[B, 1] - S[A, 1]; e1z = S[B, 2] - S[A, 2]
            e2x = S[Cc, 0] - S[A, 0]; e2y = S[Cc, 1] - S[A, 1]; e2z = S[Cc, 2] - S[A, 2]
            du1 = bx - ax; dv1 = -(by - ay); du2 = cx - ax; dv2 = -(cy - ay)
            det = du1 * dv2 - du2 * dv1
            r = 1.0 / det if det != 0 else 0.0
            Tx = (e1x * dv2 - e2x * dv1) * r; Ty = (e1y * dv2 - e2y * dv1) * r; Tz = (e1z * dv2 - e2z * dv1) * r
            Bx = (e2x * du1 - e1x * du2) * r; By = (e2y * du1 - e1y * du2) * r; Bz = (e2z * du1 - e1z * du2) * r
            hx = nfy * Tz - nfz * Ty; hy = nfz * Tx - nfx * Tz; hz = nfx * Ty - nfy * Tx
            hand = 1.0 if hx * Bx + hy * By + hz * Bz >= 0 else -1.0
            xs = max(x0, int(np.floor(min(ax, bx, cx) - dilate)))
            xe = min(x0 + cw, int(np.ceil(max(ax, bx, cx) + dilate)) + 1, W)
            ys = max(y0, int(np.floor(min(ay, by, cy) - dilate)))
            ye = min(y0 + ch, int(np.ceil(max(ay, by, cy) + dilate)) + 1, W)
            hint = -1
            for ty in range(ys, ye):
                # scanline span of the dilated triangle in this texel row (long thin triangles
                # of large flat charts would otherwise scan their whole bounding box)
                ya = ty - dilate; yb = ty + 1 + dilate
                rx0 = 1e30; rx1 = -1e30
                for ed in range(3):
                    if ed == 0:
                        x1 = ax; y1 = ay; x2 = bx; y2 = by
                    elif ed == 1:
                        x1 = bx; y1 = by; x2 = cx; y2 = cy
                    else:
                        x1 = cx; y1 = cy; x2 = ax; y2 = ay
                    lo_y = min(y1, y2); hi_y = max(y1, y2)
                    if hi_y < ya or lo_y > yb:
                        continue
                    if y2 == y1:
                        rx0 = min(rx0, x1, x2); rx1 = max(rx1, x1, x2)
                        continue
                    for yy in (max(ya, lo_y), min(yb, hi_y)):
                        xx = x1 + (x2 - x1) * (yy - y1) / (y2 - y1)
                        rx0 = min(rx0, xx); rx1 = max(rx1, xx)
                if rx1 < rx0:
                    continue
                sx = max(xs, int(np.floor(rx0 - dilate)))
                ex = min(xe, int(np.ceil(rx1 + dilate)) + 1)
                for tx in range(sx, ex):
                    rad2 = rad2f
                    d = _closest2d(tx + 0.5, ty + 0.5, ax, ay, bx, by, cx, cy, bc)
                    if d > dilate or d >= owner[ty, tx]:
                        continue
                    b0 = bc[0]; b1 = bc[1]; b2 = bc[2]
                    px = b0 * S[A, 0] + b1 * S[B, 0] + b2 * S[Cc, 0]
                    py = b0 * S[A, 1] + b1 * S[B, 1] + b2 * S[Cc, 1]
                    pz = b0 * S[A, 2] + b1 * S[B, 2] + b2 * S[Cc, 2]
                    # correspondence: the successive-mapping lineage (+ facing);
                    # nearest original surface first (fast, unfiltered): bounds the filtered
                    # searches - a lineage match much farther away than the true surface is wrong
                    # (only needed on triangles much larger than the original resolution; on
                    # small ones the lineage search is local anyway)
                    g0 = -1; d0 = 0.0
                    if dm > big_face2:
                        g0, d0 = bvh_closest(P, F, idx, box, left, start, count, px, py, pz, hint0,
                                             0, reps, A, B, Cc, ofn, nfx, nfy, nfz, stack, sd, o6, t6)
                    if g0 >= 0:
                        hint0 = g0
                        for k6 in range(6):
                            keep6[k6] = o6[k6]
                        rr = 3.0 * np.sqrt(d0) + 0.1 * np.sqrt(dm)
                        rad2 = min(rad2, rr * rr + 1e-20)
                    g, _d2 = bvh_closest(P, F, idx, box, left, start, count, px, py, pz, hint,
                                         2, reps, A, B, Cc, ofn, nfx, nfy, nfz, stack, sd, o6, t6, rad2)
                    if g < 0:
                        g, _d2 = bvh_closest(P, F, idx, box, left, start, count, px, py, pz, -1,
                                             1, reps, A, B, Cc, ofn, nfx, nfy, nfz, stack, sd, o6, t6, rad2)
                    if g < 0 and g0 >= 0:  # no lineage match near the surface: nearest point
                        g = g0
                        for k6 in range(6):
                            o6[k6] = keep6[k6]
                    if g < 0:
                        g, _d2 = bvh_closest(P, F, idx, box, left, start, count, px, py, pz, -1,
                                             0, reps, A, B, Cc, ofn, nfx, nfy, nfz, stack, sd, o6, t6)
                    if g < 0:
                        continue
                    hint = g
                    c0 = o6[0]; c1 = o6[1]; c2 = o6[2]
                    owner[ty, tx] = d
                    covered[ty, tx] = 1
                    if do_color:
                        _sample_color(g, c0, c1, c2, face_mat, corner_uv, has_uv, corner_col, has_col,
                                      mat_color, mat_tex, tex, tex_off, tex_w, tex_h, col, tmp)
                        color[ty, tx, 0] = _to_srgb8(col[0])
                        color[ty, tx, 1] = _to_srgb8(col[1])
                        color[ty, tx, 2] = _to_srgb8(col[2])
                        color[ty, tx, 3] = min(255, max(0, int(col[3] * 255.0 + 0.5)))
                    if do_mr:
                        metal, rough = _sample_mr(g, c0, c1, c2, face_mat, corner_uv, has_uv,
                                                  mat_mr, mat_mrtex, tex, tex_off, tex_w, tex_h, tmp)
                        mr[ty, tx, 0] = 255
                        mr[ty, tx, 1] = min(255, max(0, int(rough * 255.0 + 0.5)))
                        mr[ty, tx, 2] = min(255, max(0, int(metal * 255.0 + 0.5)))
                    if do_normal:
                        # original shading normal at T(p)
                        onx = c0 * onrm[g, 0, 0] + c1 * onrm[g, 1, 0] + c2 * onrm[g, 2, 0]
                        ony = c0 * onrm[g, 0, 1] + c1 * onrm[g, 1, 1] + c2 * onrm[g, 2, 1]
                        onz = c0 * onrm[g, 0, 2] + c1 * onrm[g, 1, 2] + c2 * onrm[g, 2, 2]
                        # interpolated low-poly normal + orthonormal tangent frame
                        Nx = b0 * snrm[f, 0, 0] + b1 * snrm[f, 1, 0] + b2 * snrm[f, 2, 0]
                        Ny = b0 * snrm[f, 0, 1] + b1 * snrm[f, 1, 1] + b2 * snrm[f, 2, 1]
                        Nz = b0 * snrm[f, 0, 2] + b1 * snrm[f, 1, 2] + b2 * snrm[f, 2, 2]
                        nl = np.sqrt(Nx * Nx + Ny * Ny + Nz * Nz)
                        ol = np.sqrt(onx * onx + ony * ony + onz * onz)
                        if nl > 0 and ol > 0:
                            Nx /= nl; Ny /= nl; Nz /= nl
                            onx /= ol; ony /= ol; onz /= ol
                            dt = Tx * Nx + Ty * Ny + Tz * Nz
                            tx_ = Tx - dt * Nx; ty_ = Ty - dt * Ny; tz_ = Tz - dt * Nz
                            tl = np.sqrt(tx_ * tx_ + ty_ * ty_ + tz_ * tz_)
                            if tl > 0:
                                tx_ /= tl; ty_ /= tl; tz_ /= tl
                            bx_ = hand * (Ny * tz_ - Nz * ty_)
                            by_ = hand * (Nz * tx_ - Nx * tz_)
                            bz_ = hand * (Nx * ty_ - Ny * tx_)
                            vx = onx * tx_ + ony * ty_ + onz * tz_
                            vy = onx * bx_ + ony * by_ + onz * bz_
                            vz = onx * Nx + ony * Ny + onz * Nz
                            if vz < 0.05:  # back-facing sample: clamp to the hemisphere
                                vz = 0.05
                                s2 = vx * vx + vy * vy
                                if s2 > 0:
                                    k = np.sqrt((1 - vz * vz) / s2)
                                    vx *= k; vy *= k
                            vl = np.sqrt(vx * vx + vy * vy + vz * vz)
                            normal[ty, tx, 0] = min(255, max(0, int((vx / vl * 0.5 + 0.5) * 255 + 0.5)))
                            normal[ty, tx, 1] = min(255, max(0, int((vy / vl * 0.5 + 0.5) * 255 + 0.5)))
                            normal[ty, tx, 2] = min(255, max(0, int((vz / vl * 0.5 + 0.5) * 255 + 0.5)))
                        else:
                            normal[ty, tx, 0] = 128; normal[ty, tx, 1] = 128; normal[ty, tx, 2] = 255
    return color, normal, mr, covered


# ------------------------------------------------------------------------------------
# driver
# ------------------------------------------------------------------------------------
def _pack_textures(materials):
    imgs, mat_tex, mat_mrtex = [], [], []
    for m in materials:
        for attr, lst in (("texture", mat_tex), ("mr_texture", mat_mrtex)):
            img = getattr(m, attr)
            if img is None:
                lst.append(-1)
            else:
                lst.append(len(imgs))
                imgs.append(np.ascontiguousarray(img, dtype=np.uint8))
    if not imgs:
        imgs = [np.zeros((1, 1, 4), np.uint8)]
    off = np.cumsum([0] + [im.size for im in imgs])[:-1].astype(np.int64)
    tex = np.concatenate([im.reshape(-1) for im in imgs]).astype(np.uint8)
    return (tex, off, np.array([im.shape[1] for im in imgs], np.int64),
            np.array([im.shape[0] for im in imgs], np.int64),
            np.array(mat_tex, np.int64), np.array(mat_mrtex, np.int64))


def choose_atlas_size(n_faces, requested=None):
    """Atlas big enough for ~20+ texel charts (sharp textures / normal detail)."""
    if requested:
        return int(requested)
    if n_faces <= 4000:
        return 1024
    if n_faces <= 20000:
        return 2048
    return 4096


def bake(simp_P, simp_F, prep, asset, vertex_map, atlas_size=None, bake_color=True,
         bake_normal=True, bake_mr=True, crease_angle=60.0, orig_bvh=None):
    """Returns dict with per-corner positions (normalised space), normals, uv and images."""
    S = np.ascontiguousarray(simp_P, dtype=np.float64)
    SF = np.ascontiguousarray(simp_F, dtype=np.int64)
    m = len(SF)
    crease_cos = float(np.cos(np.radians(crease_angle)))
    snrm, sfn = corner_normals(S, SF, crease_cos)

    W = choose_atlas_size(m, atlas_size)
    while True:
        lay = layout_atlas(S, SF, W)
        if lay is not None or W >= 8192:
            break
        W *= 2
    if lay is None:
        raise ValueError("too many faces for an 8192 texture atlas")
    tc, cell, item_start, item_faces, layout_info = lay

    OP, OF = prep.positions, prep.faces
    bvh = orig_bvh or BVH(OP, OF)
    reps = np.ascontiguousarray(vertex_map[OF], dtype=np.int64)
    onrm, ofn = corner_normals(OP, OF, float(np.cos(np.radians(crease_angle))))

    fo = prep.face_orig
    face_mat = np.ascontiguousarray(asset.face_material[fo], dtype=np.int64)
    has_uv = asset.corner_uv is not None
    corner_uv = np.ascontiguousarray(asset.corner_uv[fo]) if has_uv else np.zeros((1, 3, 2))
    has_col = asset.corner_color is not None
    corner_col = np.ascontiguousarray(asset.corner_color[fo]) if has_col else np.ones((1, 3, 4))
    mat_color = np.array([mm.color for mm in asset.materials], dtype=np.float64)
    mat_mr = np.array([[mm.metallic, mm.roughness] for mm in asset.materials], dtype=np.float64)
    tex, tex_off, tex_w, tex_h, mat_tex, mat_mrtex = _pack_textures(asset.materials)

    # "large" simplified triangle: longest edge above 8x the original mean edge length
    oe = np.linalg.norm(OP[OF[:, 1]] - OP[OF[:, 0]], axis=1).mean() if len(OF) else 1.0
    big_face2 = float((8.0 * oe) ** 2)
    do_color = bool(bake_color and asset.has_appearance)
    do_mr = bool(bake_mr and asset.has_mr_variation)
    color, normal, mr, covered = bake_kernel(
        W, S, SF, snrm, sfn, tc, cell, item_start, item_faces,
        *bvh.arrays, reps, ofn, onrm,
        face_mat, corner_uv, has_uv, corner_col, has_col, mat_color, mat_tex,
        mat_mr, mat_mrtex, tex, tex_off, tex_w, tex_h,
        do_color, bool(bake_normal), do_mr, float(PAD) + 0.75, big_face2)

    cov = covered.astype(bool)
    if do_color and cov.any():  # fill the unused atlas area with the mean colour (mip-friendly)
        mean = color[cov].mean(0).astype(np.uint8)
        color[~cov] = mean
    if bake_normal:
        normal[~cov] = (128, 128, 255)
    if do_mr and cov.any():
        mr[~cov] = mr[cov].mean(0).astype(np.uint8)

    uv = np.empty((m * 3, 2))
    uv[:, 0] = tc[:, :, 0].reshape(-1) / W
    uv[:, 1] = 1.0 - tc[:, :, 1].reshape(-1) / W
    base = asset.materials[0] if asset.materials else None
    return dict(
        positions=S[SF].reshape(-1, 3),
        normals=snrm.reshape(-1, 3),
        uv=uv,
        color=color if do_color else None,
        normal_map=normal if bake_normal else None,
        mr=mr if do_mr else None,
        atlas_size=W,
        **layout_info,
        base_color=tuple(base.color) if base is not None else (0.8, 0.8, 0.8, 1.0),
        metallic=base.metallic if base is not None else 0.0,
        roughness=base.roughness if base is not None else 1.0,
    )
