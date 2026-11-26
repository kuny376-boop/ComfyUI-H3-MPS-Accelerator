"""Opt-in weight-only INT8 path for resident H3 mixed precision Linear modules."""
from contextlib import contextmanager
import functools
import logging
import threading
import torch
import torch.nn.functional as F

class PackedLinear:
    def __init__(self):
        self.checked = {}
        self.hits = 0
        self.fallbacks = 0
        self.disabled = False

    def compute(self, x, q, scale, bias):
        from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation
        h = _build_hadamard(256, device=x.device, dtype=torch.float32)
        xr = _rotate_activation(x.reshape(-1, x.shape[-1]).float(), h, 256).to(x.dtype).contiguous()
        s = scale.to(dtype=x.dtype).reshape(-1)
        if s.numel() == 1:
            s = s.expand(q.shape[0])
        y = torch._weight_int8pack_mm(xr, q, s.contiguous())
        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], q.shape[0])

    def run(self, module, original, x, args, kwargs):
        from comfy_kitchen.tensor.int8 import QuantizedTensor
        w = getattr(module, "weight", None)
        p = getattr(w, "_params", None)
        q = getattr(w, "_qdata", None)
        s = getattr(p, "scale", None)
        b = getattr(module, "bias", None)
        if (self.disabled or type(x) is not torch.Tensor or x.ndim < 2
                or x.device.type != "mps" or x.dtype != torch.bfloat16 or x.requires_grad
                or not isinstance(w, QuantizedTensor)
                or getattr(w, "_layout_cls", None) != "TensorWiseINT8Layout"
                or not getattr(module, "_full_precision_mm", False)
                or getattr(module, "_full_precision_mm_config", False)
                or getattr(module, "comfy_force_cast_weights", False)
                or getattr(module, "comfy_force_forward", False)
                or getattr(module, "weight_function", None) != []
                or getattr(module, "bias_function", None) != []
                or getattr(module, "pre_quant_scale", None) is not None
                or getattr(module, "input_scale", None) is not None
                or getattr(module, "training", False)
                or not getattr(p, "convrot", False)
                or getattr(p, "convrot_groupsize", None) != 256
                or getattr(p, "transposed", True) or getattr(p, "tiled", True)
                or not isinstance(q, torch.Tensor) or q.ndim != 2
                or q.device != x.device or q.dtype != torch.int8 or not q.is_contiguous()
                or q.shape[1] != x.shape[-1] or q.shape[1] % 256 or q.shape[0] % 32
                or not isinstance(s, torch.Tensor) or s.device != x.device
                or tuple(s.shape) not in ((), (1,), (1,1), (q.shape[0],1))
                or (b is not None and (b.device != x.device or b.dtype != x.dtype))
                or args or kwargs):
            self.fallbacks += 1
            return original(x, *args, **kwargs)
        key = (id(module), id(w), q.data_ptr(), s.data_ptr())
        try:
            if key not in self.checked:
                sample = x.reshape(-1, x.shape[-1])[:16].contiguous()
                ref = original(sample).float()
                got = self.compute(sample, q, s, b).float()
                rel = ((got-ref).norm() / ref.norm().clamp_min(1e-12)).item()
                ok = bool(torch.isfinite(got).all().item()) and rel <= 0.005
                self.checked[key] = ok
                logging.info("[H3 Packed] shape=%s rel=%.6g passed=%s", tuple(q.shape), rel, ok)
            if self.checked[key]:
                y = self.compute(x, q, s, b)
                self.hits += 1
                return y
        except (RuntimeError, ValueError, TypeError, NotImplementedError) as exc:
            self.checked[key] = False
            logging.warning("[H3 Packed] native fallback: %r", exc)
        self.fallbacks += 1
        return original(x, *args, **kwargs)

@contextmanager
def packed_scope(root, engine):
    owner = threading.get_ident()
    patches = []
    try:
        for module in root.modules():
            if (not hasattr(module, "forward_comfy_cast_weights")
                    or getattr(module, "quant_format", None) != "int8_tensorwise"):
                continue
            original = module.forward
            prior = module.__dict__.get("forward")
            had = "forward" in module.__dict__
            def make(m, fn):
                @functools.wraps(fn)
                def forward(x, *args, **kwargs):
                    if threading.get_ident() != owner:
                        return fn(x, *args, **kwargs)
                    return engine.run(m, fn, x, args, kwargs)
                return forward
            replacement = make(module, original)
            module.forward = replacement
            patches.append((module, had, prior, replacement))
        yield
    finally:
        for module, had, prior, replacement in reversed(patches):
            if module.__dict__.get("forward") is replacement:
                if had:
                    module.forward = prior
                else:
                    del module.forward
