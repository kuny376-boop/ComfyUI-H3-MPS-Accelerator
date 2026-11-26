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



class H3MPSMemoryManager:
    """Opt-in conditioning barrier; never changes model precision."""
    CATEGORY = "H3 MPS Accelerator"
    FUNCTION = "apply"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "offload_text_encoder": ("BOOLEAN", {"default": True}),
        }}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def apply(self, conditioning, offload_text_encoder):
        if mm.get_torch_device().type != "mps":
            return (conditioning,)
        import gc
        import logging
        import psutil

        def snapshot():
            return {
                "allocated_MiB": round(torch.mps.current_allocated_memory() / 2**20),
                "driver_MiB": round(torch.mps.driver_allocated_memory() / 2**20),
                "available_MiB": round(psutil.virtual_memory().available / 2**20),
                "swap_MiB": round(psutil.swap_memory().used / 2**20),
            }

        torch.mps.synchronize()
        logging.info("[H3 MPS Memory] before: %s", snapshot())
        if offload_text_encoder:
            entries = getattr(mm, "current_loaded_models", None)
            if isinstance(entries, list):
                for entry in list(entries):
                    patcher = getattr(entry, "model", None)
                    model = getattr(patcher, "model", None)
                    if type(model).__name__ != "MiniMaxH3TEModel_":
                        continue
                    if getattr(entry.device, "type", None) != "mps":
                        continue
                    if not callable(getattr(entry, "model_unload", None)):
                        continue
                    if getattr(entry, "model_finalizer", None) is None:
                        continue
                    if entry.model_unload():
                        entries.remove(entry)
                        logging.info("[H3 MPS Memory] offloaded H3 text encoder")
            else:
                logging.warning("[H3 MPS Memory] unsupported model registry; skip offload")
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()
        logging.info("[H3 MPS Memory] after: %s", snapshot())
        return (conditioning,)



class H3MPSQuantAccelerator:
    """Experimental row-chunked INT8 linear; scoped to an H3 model call."""
    CATEGORY = "H3 MPS Accelerator"
    FUNCTION = "apply"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "backend": (["observe", "chunked", "convrot_metal", "convrot_direct", "native", "int8_packed"], {"default": "observe"}),
            "chunk_mib": ("INT", {"default": 128, "min": 64, "max": 512, "step": 64}),
        }}

    def apply(self, model, backend, chunk_mib):
        import functools
        import inspect
        import logging
        import threading
        import comfy_kitchen as ck
        from torch.utils._python_dispatch import TorchDispatchMode

        if mm.get_torch_device().type != "mps" or not _is_h3(model):
            return (model,)
        patched = model.clone()
        previous = patched.model_options.get("model_function_wrapper")
        lock = _H3_QUANT_LOCK
        from .convrot_metal import ConvRotMetal
        convrot = ConvRotMetal() if backend in ("convrot_metal", "convrot_direct") else None

        if backend == "native":
            return (model,)

        from .packed_linear import PackedLinear, packed_scope
        packed = PackedLinear() if backend == "int8_packed" else None

        def wrapper(apply_model, params):
            with lock:
                if packed is not None:
                    hits, misses = packed.hits, packed.fallbacks
                    try:
                        with packed_scope(patched.get_model_object("diffusion_model"), packed):
                            if previous is not None:
                                return previous(apply_model, params)
                            return apply_model(params["input"], params["timestep"], **params["c"])
                    finally:
                        logging.info("[H3 Packed] step hits=%s fallback=%s cumulative_hits=%s",
                                     packed.hits-hits, packed.fallbacks-misses, packed.hits)
                if convrot is not None:
                    # No public int8_linear monkeypatch or general operator tracing
                    # in Metal mode. Only intercept the exact ConvRot custom op.
                    class ConvRotOnly(TorchDispatchMode):
                        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                            if str(func) == "comfy_kitchen.dequantize_int8_convrot_weight_dtype.default":
                                return convrot.call(func, args, kwargs or {})
                            return func(*args, **(kwargs or {}))

                    hits, fallbacks = convrot.hits, convrot.fallbacks
                    try:
                        if backend == "convrot_direct":
                            from .convrot_direct import direct_scope
                            scope = direct_scope(convrot)
                        else:
                            scope = ConvRotOnly()
                        with scope:
                            if previous is not None:
                                return previous(apply_model, params)
                            return apply_model(params["input"], params["timestep"], **params["c"])
                    finally:
                        logging.info("[H3 ConvRot] step hits=%s fallback=%s cumulative_hits=%s disabled=%s",
                                     convrot.hits - hits, convrot.fallbacks - fallbacks,
                                     convrot.hits, convrot.disabled)

                original = getattr(ck, "int8_linear", None)
                def run():
                    if previous is not None:
                        return previous(apply_model, params)
                    return apply_model(params["input"], params["timestep"], **params["c"])
                if not callable(original):
                    logging.warning("[H3 Quant] int8_linear unavailable")
                    return run()
                try:
                    signature = inspect.signature(original)
                except (TypeError, ValueError):
                    return run()
                owner = threading.get_ident()
                counts = {"calls": 0, "chunked": 0, "skipped": 0}
                checked = {}
                dispatch_counts = {}
                reasons = {}

                class QuantObserve(TorchDispatchMode):
                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        name = str(func)
                        if convrot is not None and name == "comfy_kitchen.dequantize_int8_convrot_weight_dtype.default":
                            dispatch_counts[name] = dispatch_counts.get(name, 0) + 1
                            return convrot.call(func, args, kwargs or {})
                        if "comfy_kitchen" in name or "int_mm" in name or "scaled_mm" in name:
                            dispatch_counts[name] = dispatch_counts.get(name, 0) + 1
                        return func(*args, **(kwargs or {}))

                def skip(reason, args, kwargs):
                    counts["skipped"] += 1
                    reasons[reason] = reasons.get(reason, 0) + 1
                    return original(*args, **kwargs)

                @functools.wraps(original)
                def linear(*args, **kwargs):
                    if threading.get_ident() != owner:
                        return original(*args, **kwargs)
                    counts["calls"] += 1
                    bound = signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    a = bound.arguments
                    x, w = a.get("x"), a.get("weight")
                    if counts["calls"] <= 4:
                        logging.info("[H3 Quant] call %s x=%s weight=%s act=%s",
                                     counts["calls"], getattr(x, "shape", None),
                                     getattr(w, "shape", None), a.get("input_act"))
                    if (backend != "chunked" or not isinstance(x, torch.Tensor)
                            or not isinstance(w, torch.Tensor) or x.device.type != "mps"
                            or x.ndim != 2 or w.ndim != 2 or w.dtype != torch.int8
                            or a.get("weight_tiled", False) or a.get("residual") is not None
                            or a.get("residual_scale") is not None
                            or a.get("input_act") not in (None, "swiglu", "rms_norm")
                            or w.shape[0] == 0):
                        return skip("observe_or_unsupported", args, kwargs)
                    rows = max(1, (chunk_mib << 20) // (4 * w.shape[0]))
                    if x.shape[0] <= rows:
                        return skip("below_budget", args, kwargs)

                    def part(lo, hi):
                        call = signature.bind(*args, **kwargs)
                        call.arguments["x"] = x[lo:hi]
                        return original(*call.args, **call.kwargs)

                    key = (tuple(w.shape), x.dtype, a.get("out_dtype"),
                           a.get("input_act"), a.get("convrot"), a.get("convrot_groupsize"))
                    if key not in checked:
                        # Compare a small native projection against two row chunks.
                        ref = part(0, 32)
                        probe = torch.cat((part(0, 16), part(16, 32)), dim=0)
                        checked[key] = bool(torch.equal(ref, probe))
                        logging.info("[H3 Quant] exact small-row check %s: %s", key, checked[key])
                        del ref, probe
                    if not checked[key]:
                        return skip("numeric_check", args, kwargs)
                    out = None
                    for lo in range(0, x.shape[0], rows):
                        hi = min(lo + rows, x.shape[0])
                        block = part(lo, hi)
                        if out is None:
                            out = torch.empty((x.shape[0], block.shape[-1]),
                                              dtype=block.dtype, device=block.device)
                        out[lo:hi].copy_(block)
                        del block
                    counts["chunked"] += 1
                    return out

                ck.int8_linear = linear
                try:
                    with QuantObserve():
                        return run()
                finally:
                    if ck.int8_linear is linear:
                        ck.int8_linear = original
                    logging.info("[H3 Quant] backend=%s chunk_mib=%s %s",
                                 backend, chunk_mib, counts)
                    logging.info("[H3 Quant] dispatch=%s skipped_reasons=%s", dispatch_counts, reasons)
                    if convrot is not None:
                        logging.info("[H3 ConvRot] cumulative hits=%s fallback=%s disabled=%s", convrot.hits, convrot.fallbacks, convrot.disabled)

        patched.set_model_unet_function_wrapper(wrapper)
        return (patched,)


import threading as _threading
_H3_QUANT_LOCK = _threading.RLock()

NODE_CLASS_MAPPINGS = {
    "H3MPSQuantAccelerator": H3MPSQuantAccelerator,
    "H3MPSMemoryManager": H3MPSMemoryManager,
    "H3MPSRuntimeGuard": H3MPSRuntimeGuard,
    "H3MPSAttentionProfiler": H3MPSAttentionProfiler,
    "H3MPSProfilerReport": H3MPSProfilerReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3MPSQuantAccelerator": "H3 MPS Quant Accelerator",
    "H3MPSMemoryManager": "H3 MPS Memory Manager",
    "H3MPSRuntimeGuard": "H3 MPS Runtime Guard",
    "H3MPSAttentionProfiler": "H3 MPS Attention Profiler",
    "H3MPSProfilerReport": "H3 MPS Profiler Report",
}

