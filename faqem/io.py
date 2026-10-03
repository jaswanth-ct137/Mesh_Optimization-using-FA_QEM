"""Loading / saving meshes with their appearance (textures, vertex colours, materials)."""
from dataclasses import dataclass, field

import numpy as np
import trimesh
from PIL import Image

SUPPORTED = (".glb", ".gltf", ".obj", ".stl", ".ply", ".off", ".3mf", ".dae")


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


@dataclass
class Material:
    color: np.ndarray                 # linear RGBA factor
    texture: np.ndarray = None        # (h, w, 4) uint8 sRGB, or None
    metallic: float = 0.0
    roughness: float = 1.0
    mr_texture: np.ndarray = None     # (h, w, 4) uint8 (glTF: G = roughness, B = metallic)
    alpha_mode: str = "OPAQUE"
    double_sided: bool = False


@dataclass
class Asset:
    positions: np.ndarray             # (n, 3) model space
    faces: np.ndarray                 # (m, 3)
    face_material: np.ndarray         # (m,)
    materials: list
    corner_uv: np.ndarray = None      # (m, 3, 2) OpenGL convention (v up)
    corner_color: np.ndarray = None   # (m, 3, 4) linear RGBA
    info: dict = field(default_factory=dict)

    @property
    def has_texture(self):
        return any(m.texture is not None for m in self.materials) and self.corner_uv is not None

    @property
    def has_appearance(self):
        """True when colour varies over the surface (texture, vertex colours or several materials)."""
        if self.has_texture or self.corner_color is not None:
            return True
        cols = {tuple(np.round(m.color, 4)) for m in self.materials}
        return len(cols) > 1

    @property
    def has_mr_variation(self):
        if any(m.mr_texture is not None for m in self.materials) and self.corner_uv is not None:
            return True
        return len({(round(m.metallic, 4), round(m.roughness, 4)) for m in self.materials}) > 1


def _img_rgba(img):
    if img is None:
        return None
    if not isinstance(img, Image.Image):
        try:
            img = Image.fromarray(np.asarray(img))
        except Exception:
            return None
    return np.ascontiguousarray(np.asarray(img.convert("RGBA")))


def _material_from_visual(visual):
    mat = getattr(visual, "material", None)
    if mat is None:
        return Material(color=np.array([0.8, 0.8, 0.8, 1.0]))
    if isinstance(mat, trimesh.visual.material.SimpleMaterial):
        mat = mat.to_pbr()
    bcf = getattr(mat, "baseColorFactor", None)
    if bcf is None:
        color = np.ones(4)
    else:
        bcf = np.asarray(bcf, dtype=np.float64)
        color = bcf / 255.0 if bcf.max() > 1.0 or bcf.dtype == np.uint8 else bcf
        if len(color) == 3:
            color = np.append(color, 1.0)
    return Material(
        color=color,
        texture=_img_rgba(getattr(mat, "baseColorTexture", None)),
        metallic=float(mat.metallicFactor if getattr(mat, "metallicFactor", None) is not None else 0.0),
        roughness=float(mat.roughnessFactor if getattr(mat, "roughnessFactor", None) is not None else 1.0),
        mr_texture=_img_rgba(getattr(mat, "metallicRoughnessTexture", None)),
        alpha_mode=str(getattr(mat, "alphaMode", None) or "OPAQUE"),
        double_sided=bool(getattr(mat, "doubleSided", False) or False),
    )


def load_asset(path):
    """Load any supported mesh file into a flat Asset (scene transforms applied)."""
    scene = trimesh.load(path, force="scene", process=False)
    pos, faces, fmat, uvs, cols = [], [], [], [], []
    materials, mat_keys = [], {}
    has_uv = has_col = False
    parts = []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname]
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        parts.append((T, g))
    if not parts:
        raise ValueError("No triangle mesh found in the file.")
    off = 0
    for T, g in parts:
        V = np.asarray(g.vertices, dtype=np.float64)
        Fc = np.asarray(g.faces, dtype=np.int64)
        V = V @ T[:3, :3].T + T[:3, 3]
        flip = np.linalg.det(T[:3, :3]) < 0
        if flip:
            Fc = Fc[:, ::-1]
        vis = g.visual
        key = id(getattr(vis, "material", None))
        if key not in mat_keys:
            mat_keys[key] = len(materials)
            materials.append(_material_from_visual(vis))
        mid = mat_keys[key]
        # per-corner attributes
        uv = getattr(vis, "uv", None)
        if uv is not None and len(uv) == len(V):
            uvs.append(np.asarray(uv, dtype=np.float64)[Fc]); has_uv = True
        else:
            uvs.append(None)
        vc = None
        if isinstance(vis, trimesh.visual.ColorVisuals) and vis.kind == "vertex":
            vc = np.asarray(vis.vertex_colors)
        elif isinstance(vis, trimesh.visual.ColorVisuals) and vis.kind == "face":
            fc = np.asarray(vis.face_colors)
            vc_c = np.repeat(fc[:, None, :], 3, axis=1)
            cols.append(np.concatenate([srgb_to_linear(vc_c[..., :3] / 255.0), vc_c[..., 3:] / 255.0], -1))
            has_col = True
        elif hasattr(vis, "vertex_attributes") and "color" in getattr(vis, "vertex_attributes", {}):
            vc = np.asarray(vis.vertex_attributes["color"])
        if vc is not None and len(vc) == len(V):
            c = vc[Fc].astype(np.float64)
            if c.max() > 1.0:
                c = c / 255.0
            if c.shape[-1] == 3:
                c = np.concatenate([c, np.ones(c.shape[:-1] + (1,))], -1)
            cols.append(np.concatenate([srgb_to_linear(c[..., :3]), c[..., 3:]], -1)); has_col = True
        elif not (isinstance(vis, trimesh.visual.ColorVisuals) and vis.kind == "face"):
            cols.append(None)
        pos.append(V)
        faces.append(Fc + off)
        fmat.append(np.full(len(Fc), mid, np.int64))
        off += len(V)

    m = sum(len(f) for f in faces)
    def cat(lst, shape, fill):
        out = np.empty((m,) + shape)
        o = 0
        for arr, f in zip(lst, faces):
            k = len(f)
            out[o:o + k] = fill if arr is None else arr
            o += k
        return out

    asset = Asset(
        positions=np.concatenate(pos),
        faces=np.concatenate(faces),
        face_material=np.concatenate(fmat),
        materials=materials,
        corner_uv=cat(uvs, (3, 2), 0.0) if has_uv else None,
        corner_color=cat(cols, (3, 4), 1.0) if has_col else None,
    )
    asset.info = dict(parts=len(parts), materials=len(materials),
                      textured=asset.has_texture, vertex_colors=has_col)
    return asset


# ------------------------------------------------------------------------------------
# export
# ------------------------------------------------------------------------------------
def export_textured(path, positions, normals, uv, color_img=None, normal_img=None, mr_img=None,
                    base_color=(1, 1, 1, 1), metallic=0.0, roughness=1.0, double_sided=False,
                    alpha_mode="OPAQUE"):
    """Unshared-corner mesh (one atlas chart per triangle) with baked PBR textures -> GLB/OBJ."""
    n = len(positions)
    faces = np.arange(n, dtype=np.int64).reshape(-1, 3)
    mat = trimesh.visual.material.PBRMaterial(
        name="faqem_baked",
        baseColorFactor=np.round(np.asarray(base_color) * 255).astype(np.uint8) if color_img is None
        else np.array([255, 255, 255, 255], np.uint8),
        baseColorTexture=None if color_img is None else Image.fromarray(color_img),
        normalTexture=None if normal_img is None else Image.fromarray(normal_img),
        metallicRoughnessTexture=None if mr_img is None else Image.fromarray(mr_img),
        metallicFactor=1.0 if mr_img is not None else float(metallic),
        roughnessFactor=1.0 if mr_img is not None else float(roughness),
        doubleSided=bool(double_sided),
        alphaMode=alpha_mode if alpha_mode in ("OPAQUE", "MASK", "BLEND") else "OPAQUE",
    )
    mesh = trimesh.Trimesh(vertices=positions, faces=faces, vertex_normals=normals, process=False,
                           visual=trimesh.visual.TextureVisuals(uv=uv, material=mat))
    mesh.export(path, include_normals=True) if str(path).endswith((".glb", ".gltf")) else mesh.export(path)
    return path


def export_geometry(path, positions, faces, normals=None, color=(0.55, 0.56, 0.6, 1.0)):
    mesh = trimesh.Trimesh(vertices=positions, faces=faces, process=False)
    if normals is not None:
        mesh.vertex_normals = normals
    if str(path).endswith((".glb", ".gltf")):
        mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=np.round(np.asarray(color) * 255).astype(np.uint8), metallicFactor=0.0,
            roughnessFactor=0.8, doubleSided=True))
        mesh.export(path, include_normals=True)
    else:
        mesh.export(path)
    return path

