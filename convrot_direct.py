"""Experimental direct layout hook, active only during the owning model call."""
from contextlib import contextmanager
import logging
import threading
import torch

@contextmanager
def direct_scope(convrot):
    try:
        from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout, _row_major, _INT8_DEQUANT_DTYPE_TO_CODE
        cls = TensorWiseINT8Layout
        descriptor = cls.__dict__["dequantize"]
        if not isinstance(descriptor, classmethod):
            raise TypeError("unexpected dequantize descriptor")
        original = cls.dequantize
        op = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype.default
    except (ImportError, AttributeError, KeyError, TypeError) as exc:
        logging.warning("[H3 ConvRot direct] unavailable: %r", exc)
        yield
        return
    owner = threading.get_ident()

    def dequantize(layout, qdata, params):
        if (threading.get_ident() != owner or layout is not cls
                or qdata.device.type != "mps"
                or not getattr(params, "convrot", False)
                or params.orig_dtype != torch.bfloat16):
            return original(qdata, params)
        q = _row_major(qdata, params)
        code = _INT8_DEQUANT_DTYPE_TO_CODE.get(params.orig_dtype, 0)
        return convrot.call(op, (q, params.scale, params.convrot_groupsize, code), {}).to(params.orig_dtype)

    replacement = classmethod(dequantize)
    cls.dequantize = replacement
    try:
        yield
    finally:
        if cls.__dict__.get("dequantize") is replacement:
            cls.dequantize = descriptor
