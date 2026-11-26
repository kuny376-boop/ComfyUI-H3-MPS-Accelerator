"""Read-only discovery of the existing MPS INT8 backend; never changes registry priority."""
import inspect
import torch

def runtime_status():
    result = {"torch": str(torch.__version__), "mps": torch.backends.mps.is_available(),
              "native_operator": False, "external_registered": False}
    try:
        result["native_operator"] = torch._C._dispatch_has_kernel_for_dispatch_key(
            "aten::_weight_int8pack_mm", "MPS")
    except (AttributeError, RuntimeError) as exc:
        result["operator_error"] = str(exc)
    try:
        import comfy_kitchen as ck
        from comfy_kitchen.registry import registry
        result["external_registered"] = bool(registry.is_available("mps_int8_native"))
        result["priority"] = list(getattr(registry, "_priority", []))
        result["public_signature"] = str(inspect.signature(ck.int8_linear))
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
        result["registry_error"] = str(exc)
    return result
