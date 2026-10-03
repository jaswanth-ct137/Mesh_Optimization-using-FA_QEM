"""Wireframe overlays: append a mesh's edges to a GLB as one glTF LINES primitive.

Writing the line geometry straight into the binary chunk with NumPy keeps this fast even for
millions of edges (trimesh builds one Python object per line segment).
"""
import json
import struct

import numpy as np


def mesh_edges(F):
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    e.sort(axis=1)
    return np.unique(e, axis=0)


def _read_glb(data):
    magic, version, _ = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError("not a GLB file")
    off, js, binary = 12, None, b""
    while off < len(data):
        ln, typ = struct.unpack_from("<II", data, off)
        chunk = data[off + 8: off + 8 + ln]
        if typ == 0x4E4F534A:
            js = json.loads(chunk.decode("utf-8"))
        elif typ == 0x004E4942:
            binary = bytes(chunk)
        off += 8 + ln
    return js, binary


def _write_glb(js, binary):
    jb = json.dumps(js, separators=(",", ":")).encode("utf-8")
    jb += b" " * (-len(jb) % 4)
    binary += b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(jb) + 8 + len(binary)
    return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(jb), 0x4E4F534A) + jb
            + struct.pack("<II", len(binary), 0x004E4942) + binary)


def auto_alpha(P, E, viewer_px=650.0):
    """Line opacity from the on-screen edge spacing when the whole model fills the viewer:
    edges closer than ~6 px would merge into a solid colour, so they are drawn translucent
    (dense regions then read as a tint and individual triangles appear when zooming in)."""
    diag = float(np.linalg.norm(P.max(0) - P.min(0))) or 1.0
    # median, not mean: adaptive low-poly meshes have a few huge triangles and many small ones,
    # and it is the small (dense) ones that saturate the image
    typical_edge = float(np.median(np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)))
    spacing_px = typical_edge / diag * viewer_px
    return float(np.clip(spacing_px / 6.0, 0.12, 1.0))


def add_wireframe(glb_in, P, F, glb_out, color=(0.0, 0.75, 1.0, 1.0), lift=2e-4, alpha=None):
    """Copy glb_in to glb_out with the edges of (P, F) (world coordinates) drawn as lines.
    glb_in=None writes the lines alone (a pure wireframe model).

    Edges are nudged along the vertex normals by `lift` x bbox diagonal so they sit just above
    the surface instead of z-fighting with it.
    """
    P = np.asarray(P, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    if glb_in is None:
        js, binary = {"asset": {"version": "2.0", "generator": "faqem"}, "scene": 0,
                      "scenes": [{"nodes": []}], "buffers": [{"byteLength": 0}]}, b""
        lift = 0.0
    else:
        with open(glb_in, "rb") as fh:
            js, binary = _read_glb(fh.read())

    # area-weighted vertex normals for the lift
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    vn = np.zeros_like(P)
    for k in range(3):
        np.add.at(vn, F[:, k], n)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-30)
    diag = float(np.linalg.norm(P.max(0) - P.min(0))) or 1.0
    V = (P + vn * (lift * diag)).astype(np.float32)
    E = mesh_edges(F).astype(np.uint32)
    if alpha is None:
        alpha = auto_alpha(P, E)
    color = (float(color[0]), float(color[1]), float(color[2]), float(alpha))

    binary = bytearray(binary)
    binary += b"\0" * (-len(binary) % 4)
    buffers = js.setdefault("buffers", [{"byteLength": 0}])
    views = js.setdefault("bufferViews", [])
    accessors = js.setdefault("accessors", [])

    def append(arr, target):
        start = len(binary)
        binary.extend(arr.tobytes())
        binary.extend(b"\0" * (-len(binary) % 4))
        views.append({"buffer": 0, "byteOffset": start, "byteLength": arr.nbytes, "target": target})
        return len(views) - 1

    pv = append(V, 34962)
    accessors.append({"bufferView": pv, "componentType": 5126, "count": int(len(V)), "type": "VEC3",
                      "min": V.min(0).tolist(), "max": V.max(0).tolist()})
    pos_acc = len(accessors) - 1
    iv = append(E.reshape(-1), 34963)
    accessors.append({"bufferView": iv, "componentType": 5125, "count": int(E.size), "type": "SCALAR"})
    idx_acc = len(accessors) - 1

    mats = js.setdefault("materials", [])
    mat = {"name": "wireframe", "pbrMetallicRoughness": {"baseColorFactor": list(color),
           "metallicFactor": 0.0, "roughnessFactor": 1.0}, "extensions": {"KHR_materials_unlit": {}}}
    if color[3] < 1.0:
        mat["alphaMode"] = "BLEND"
    mats.append(mat)
    used = js.setdefault("extensionsUsed", [])
    if "KHR_materials_unlit" not in used:
        used.append("KHR_materials_unlit")
    meshes = js.setdefault("meshes", [])
    meshes.append({"name": "wireframe", "primitives": [
        {"attributes": {"POSITION": pos_acc}, "indices": idx_acc, "mode": 1, "material": len(mats) - 1}]})
    nodes = js.setdefault("nodes", [])
    nodes.append({"name": "wireframe", "mesh": len(meshes) - 1})
    scenes = js.setdefault("scenes", [{"nodes": []}])
    scenes[js.get("scene", 0)].setdefault("nodes", []).append(len(nodes) - 1)
    buffers[0]["byteLength"] = len(binary)
    buffers[0].pop("uri", None)

    with open(glb_out, "wb") as fh:
        fh.write(_write_glb(js, bytes(binary)))
    return glb_out
