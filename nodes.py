"""ComfyUI node definitions."""

from __future__ import annotations

import sys
import time

import torch

import comfy.model_management as mm

from .guard import UNSAFE_FP16_FLAGS, validate_runtime
from .profiler import make_profile_override, report_session


class AnyType(str):
    def __ne__(self, _value):
        return False


ANY = AnyType("*")


def _is_h3(model) -> bool:
    diffusion_model = model.get_model_object("diffusion_model")
    return hasattr(diffusion_model, "rope_freqs") and hasattr(diffusion_model, "_forward")


def _enabled_comfy_flags() -> tuple[str, ...]:
    try:
        from comfy.cli_args import args
    except Exception:
        return ()
    mapping = (
        ("force_fp16", "--force-fp16"),
        ("fp16_unet", "--fp16-unet"),
        ("fp16_text_enc", "--fp16-text-enc"),
    )
    return tuple(flag for attr, flag in mapping if bool(getattr(args, attr, False)))


class H3MPSRuntimeGuard:
    CATEGORY = "H3 MPS Accelerator"
    FUNCTION = "check"
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "strict": ("BOOLEAN", {"default": True}),
            }
        }

    def check(self, model, strict):
        device = mm.get_torch_device()
        result = validate_runtime(
            platform=sys.platform,
            device=device.type,
            is_h3=_is_h3(model),
            pytorch_attention=bool(mm.pytorch_attention_enabled()),
            enabled_flags=_enabled_comfy_flags(),
        )
        report = result.render()
        print("[H3 MPS Accelerator] " + report.replace("\n", " | "))
        if strict and not result.ok:
            raise RuntimeError(report)
        return model, report


class H3MPSAttentionProfiler:
    CATEGORY = "H3 MPS Accelerator"
    FUNCTION = "apply"
    RETURN_TYPES = ("MODEL", "H3_MPS_PROFILE")
    RETURN_NAMES = ("model", "profile")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "session_name": ("STRING", {"default": "h3_mps"}),
                "exact_mps_timing": ("BOOLEAN", {"default": False}),
            }
        }

    def apply(self, model, session_name, exact_mps_timing):
        name = session_name.strip() or f"h3_mps_{time.time_ns()}"
        patched = model.clone()
        options = patched.model_options.setdefault("transformer_options", {})
        previous = options.get("optimized_attention_override")
        sync = None
        if exact_mps_timing and mm.get_torch_device().type == "mps":
            sync = torch.mps.synchronize
        memory_probe = None
        if mm.get_torch_device().type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            memory_probe = torch.mps.current_allocated_memory
        options["optimized_attention_override"] = make_profile_override(
            session=name,
            previous=previous,
            synchronize=sync,
            memory_probe=memory_probe,
        )
        return patched, name


class H3MPSProfilerReport:
    CATEGORY = "H3 MPS Accelerator"
    FUNCTION = "report"
    OUTPUT_NODE = True
    RETURN_TYPES = (ANY, "STRING")
    RETURN_NAMES = ("signal", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": (ANY,),
                "profile": ("H3_MPS_PROFILE",),
            }
        }

    def report(self, signal, profile):
        report = report_session(profile)
        print("[H3 MPS Accelerator] attention profile\n" + report)
        return signal, report


NODE_CLASS_MAPPINGS = {
    "H3MPSRuntimeGuard": H3MPSRuntimeGuard,
    "H3MPSAttentionProfiler": H3MPSAttentionProfiler,
    "H3MPSProfilerReport": H3MPSProfilerReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3MPSRuntimeGuard": "H3 MPS Runtime Guard",
    "H3MPSAttentionProfiler": "H3 MPS Attention Profiler",
    "H3MPSProfilerReport": "H3 MPS Profiler Report",
}

