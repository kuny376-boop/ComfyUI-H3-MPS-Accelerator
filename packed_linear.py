"""Opt-in weight-only INT8 path for resident H3 mixed precision Linear modules."""
from contextlib import contextmanager
import functools
import inspect
import logging
import threading
from collections import Counter
import torch
import torch.nn.functional as F

class PackedLinear:
    def __init__(self):
        self.checked = {}
        self.hits = 0
        self.fallbacks = 0
        self.disabled = False
        self.reasons = Counter()
        self.hadamard = {}

    def compute(self, x, q, scale, bias):
        from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation
        key = (x.device, torch.float32, 256)
        h = self.hadamard.get(key)
        if h is None:
            h = _build_hadamard(256, device=x.device, dtype=torch.float32)
            self.hadamard[key] = h
        xr = _rotate_activation(x.reshape(-1, x.shape[-1]).float(), h, 256).to(x.dtype).contiguous()
        s = scale.to(dtype=x.dtype).reshape(-1)
        if s.numel() == 1:
            s = s.expand(q.shape[0])
        y = torch._weight_int8pack_mm(xr, q, s.contiguous())
        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], q.shape[0])

    def run(self, module, original, x, args, kwargs):
        # Accept the native forward's explicit default cast arguments.
        try:
            bound = inspect.signature(original).bind(x, *args, **kwargs)
            bound.apply_defaults()
        except (TypeError, ValueError):
            return original(x, *args, **kwargs)
        options = dict(bound.arguments)
        options.pop(next(iter(options)), None)
        if any(k not in ("compute_dtype", "want_requant", "weight_only_quant")
               for k in options) or options.get("want_requant") or options.get("weight_only_quant") or options.get("compute_dtype") not in (None, x.dtype):
            self.fallbacks += 1
            self.reasons["cast_options"] += 1
            return original(x, *args, **kwargs)
        if getattr(module, "comfy_cast_weights", False) or getattr(module, "_prefetch", None) is not None or getattr(module, "_v", None) is not None:
            self.fallbacks += 1
            self.reasons["managed_weight"] += 1
            return original(x, *args, **kwargs)
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
                or tuple(s.shape) not in ((), (1,), (1,1), (q.shape[0],), (q.shape[0],1))
                or (b is not None and (b.device != x.device or b.dtype != x.dtype))
                ):
            self.fallbacks += 1
            self.reasons['ineligible'] += 1
            return original(x, *args, **kwargs)
        key = (id(module), id(w), q.data_ptr(), s.data_ptr())
        try:
            if key not in self.checked:
                sample = x.reshape(-1, x.shape[-1])[:16].contiguous()
                ref = original(sample, *args, **kwargs).float()
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
            self.reasons['native_exception'] += 1
            logging.warning("[H3 Packed] native fallback: %r", exc)
        self.fallbacks += 1
        self.reasons['validation_or_native_failure'] += 1
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
            original = module.forward_comfy_cast_weights
            try:
                inspect.signature(original).bind(object())
            except (TypeError, ValueError):
                continue
            prior = module.__dict__.get("forward_comfy_cast_weights")
            had = "forward_comfy_cast_weights" in module.__dict__
            def make(m, fn):
                @functools.wraps(fn)
                def forward(x, *args, **kwargs):
                    if threading.get_ident() != owner:
                        return fn(x, *args, **kwargs)
                    return engine.run(m, fn, x, args, kwargs)
                return forward
            replacement = make(module, original)
            module.forward_comfy_cast_weights = replacement
            patches.append((module, had, prior, replacement))
        yield
    finally:
        for module, had, prior, replacement in reversed(patches):
            if module.__dict__.get("forward_comfy_cast_weights") is replacement:
                if had:
                    module.forward_comfy_cast_weights = prior
                else:
                    del module.forward_comfy_cast_weights
