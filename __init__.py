"""ComfyUI H3 MPS Accelerator.

The package is intentionally independent from ComfyUI-Banzhang-All.  It
patches an individual MODEL object and never edits ComfyUI or another custom
node in place.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

