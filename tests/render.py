"""Tiny software renderer (Numba z-buffer) for side-by-side visual checks with one camera."""
import numpy as np
from numba import njit
from PIL import Image


@njit(cache=True)
def _raster(Pc, F, fcol, W, H, img, zb, wire, wire_col):
    for f in range(F.shape[0]):
        a = F[f, 0]; b = F[f, 1]; c = F[f, 2]
        x0 = Pc[a, 0]; y0 = Pc[a, 1]; z0 = Pc[a, 2]
        x1 = Pc[b, 0]; y1 = Pc[b, 1]; z1 = Pc[b, 2]
        x2 = Pc[c, 0]; y2 = Pc[c, 1]; z2 = Pc[c, 2]
        if z0 <= 0 or z1 <= 0 or z2 <= 0:
            continue
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if area == 0:
            continue
        xs = max(0, int(min(x0, x1, x2))); xe = min(W - 1, int(max(x0, x1, x2)) + 1)
        ys = max(0, int(min(y0, y1, y2))); ye = min(H - 1, int(max(y0, y1, y2)) + 1)
        for y in range(ys, ye + 1):
            for x in range(xs, xe + 1):
                px = x + 0.5; py = y + 0.5
                w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / area
                w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / area
                w2 = 1 - w0 - w1
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                z = w0 * z0 + w1 * z1 + w2 * z2
                if z >= zb[y, x]:
                    continue
                zb[y, x] = z
                edge = False
                if wire:
                    # distance (in pixels) to the nearest edge
                    l0 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    l1 = np.sqrt((x0 - x2) ** 2 + (y0 - y2) ** 2)
                    l2 = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                    d = min(w0 * abs(area) / max(l0, 1e-9), w1 * abs(area) / max(l1, 1e-9),
                            w2 * abs(area) / max(l2, 1e-9))
                    edge = d < 0.6
                for k in range(3):
                    img[y, x, k] = wire_col[k] if edge else fcol[f, k]


def render(P, F, eye, target=None, up=(0, 1, 0), fov=35.0, size=(900, 900), wire=False,
           color=(0.72, 0.72, 0.76)):
    P = np.asarray(P, float); F = np.asarray(F, np.int64)
    target = np.asarray(P.mean(0) if target is None else target, float)
    eye = np.asarray(eye, float)
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    upv = np.cross(right, fwd)
    rel = P - eye
    cam = np.stack([rel @ right, rel @ upv, rel @ fwd], 1)
    W, H = size
    f = 0.5 * H / np.tan(np.radians(fov) / 2)
    Pc = np.stack([W / 2 + f * cam[:, 0] / cam[:, 2], H / 2 - f * cam[:, 1] / cam[:, 2], cam[:, 2]], 1)
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
    L1 = -fwd * 0.6 + upv * 0.6 + right * 0.3; L1 /= np.linalg.norm(L1)
    L2 = -fwd * 0.5 - right * 0.6; L2 /= np.linalg.norm(L2)
    view = -fwd
    ndv = n @ view
    n = np.where(ndv[:, None] < 0, -n, n)  # two-sided
    shade = 0.18 + 0.62 * np.clip(n @ L1, 0, 1) + 0.25 * np.clip(n @ L2, 0, 1)
    fcol = np.clip(shade[:, None] * np.asarray(color)[None], 0, 1)
    img = np.ones((H, W, 3))
    zb = np.full((H, W), np.inf)
    _raster(Pc, F, fcol, W, H, img, zb, wire, np.array([0.1, 0.1, 0.15]))
    return Image.fromarray((img * 255).astype(np.uint8))


def side_by_side(images, labels=None):
    W = sum(i.width for i in images); H = max(i.height for i in images)
    out = Image.new("RGB", (W, H), "white")
    x = 0
    for i in images:
        out.paste(i, (x, 0)); x += i.width
    return out
