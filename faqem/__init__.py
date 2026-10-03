"""FA-QEM: Feature-Aware Quadric Error Metric mesh simplification (arXiv:2605.14029)."""
from .core import DEFAULTS, PAPER_TABLE1, PRESETS, simplify
from .prep import prepare_mesh, mesh_stats

__all__ = ["DEFAULTS", "PAPER_TABLE1", "PRESETS", "simplify", "prepare_mesh", "mesh_stats"]
