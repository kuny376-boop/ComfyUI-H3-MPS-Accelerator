"""General-purpose MiniMax H3 acceleration and profiling for ComfyUI.

The package patches an individual MODEL object and never edits ComfyUI or
another custom node in place.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

