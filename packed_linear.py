"""Experimental weight-only INT8 path for prepared H3 mixed precision Linear modules."""
from contextlib import contextmanager
import functools
import inspect
import logging
import os
import threading
import weakref
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
        self.sample_validation = os.environ.get('H3_PACKED_SAMPLE_VALIDATION', '0') == '1'

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
        from comfy_kitchen.tensor.int8 import QuantizedTensor
        from comfy import ops
        w = getattr(module, "weight", None)
        if (not isinstance(w, QuantizedTensor)
                or not getattr(module, "_full_precision_mm", False)
                or getattr(module, "_full_precision_mm_config", False)
                or getattr(module, "comfy_force_cast_weights", False)
                or getattr(module, "comfy_force_forward", False)
                or getattr(module, "weight_function", None) != []
                or getattr(module, "bias_function", None) != []
                or getattr(module, "training", False)
                or getattr(module, "pre_quant_scale", None) is not None
                or getattr(module, "input_scale", None) is not None
                or type(x) is not torch.Tensor or x.device.type != "mps"
                or x.dtype != torch.bfloat16 or x.requires_grad):
            self.fallbacks += 1
            self.reasons["module_guard"] += 1
            return original(x, *args, **kwargs)
        if self.disabled or not hasattr(torch, "_weight_int8pack_mm"):
            self.fallbacks += 1
            self.reasons["native_unavailable"] += 1
            return original(x, *args, **kwargs)
        context = getattr(ops, "CastBiasWeightContext", None)
        if context is None:
            self.reasons["missing_context"] += 1
            self.fallbacks += 1
            return original(x, *args, **kwargs)
        try:
            inspect.signature(ops.cast_bias_weight).bind(module, input=None,
                dtype=w.dtype, device=x.device, bias_dtype=x.dtype,
                offloadable=True, compute_dtype=x.dtype, want_requant=True)
        except (AttributeError, TypeError, ValueError):
            self.reasons["context_signature"] += 1
            self.fallbacks += 1
            return original(x, *args, **kwargs)
        with context(module, input=None, dtype=w.dtype, device=x.device,
                     bias_dtype=x.dtype, offloadable=True,
                     compute_dtype=x.dtype, want_requant=True) as (prepared, bias):
            result = self.run_prepared(module, x, prepared, bias)
            if result is not None:
                return result
        return original(x, *args, **kwargs)

    def run_prepared(self, module, x, w, b):
        from comfy_kitchen.tensor.int8 import QuantizedTensor
        p = getattr(w, "_params", None)
        q = getattr(w, "_qdata", None)
        s = getattr(p, "scale", None)
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
            return None
        # Reuse validation only for unchanged, live source buffers. Never trust
        # a recycled prepared-buffer address or an untracked inference tensor.
        source = getattr(module, 'weight', None)
        sq = getattr(source, '_qdata', None)
        ss = getattr(getattr(source, '_params', None), 'scale', None)
        key = None
        try:
            if sq is q and ss is s and b is None:
                key = (id(module), id(q), id(s), q._version, s._version,
                       tuple(q.shape), tuple(s.shape), x.dtype, x.device)
        except (RuntimeError, AttributeError):
            pass
        entry = self.checked.get(id(module)) if key is not None else None
        cached = (entry is not None and entry[0] == key
                  and entry[1]() is q and entry[2]() is s)
        ok = entry[3] if cached else False
        try:
            if not cached:
                sample = x.reshape(-1, x.shape[-1])[:16].contiguous()
                if self.sample_validation:
                    # Three disjoint 128-channel bands, preserving native alignment.
                    from comfy_kitchen.backends.eager.quantization import dequantize_int8_convrot_weight
                    n = q.shape[0]
                    width = min(128, n)
                    starts = sorted({0, ((n-width)//64)*32, n-width})
                    idx = torch.cat([torch.arange(a, a+width, device=q.device) for a in starts]).unique(sorted=True)
                    qs = q.index_select(0, idx).contiguous()
                    ssample = s.reshape(-1)
                    if ssample.numel() != 1:
                        ssample = ssample.index_select(0, idx).reshape(-1, 1)
                    bs = b.index_select(0, idx) if b is not None else None
                    ref = F.linear(sample, dequantize_int8_convrot_weight(qs, ssample, 256).to(sample.dtype), bs).float()
                    # Check the actual full-width native kernel, not a different small GEMM.
                    got = self.compute(sample, q, s, b).index_select(-1, idx).float()
                    self.reasons['sampled_validation'] += 1
                else:
                    ref = F.linear(sample, w.dequantize().to(sample.dtype), b).float()
                    got = self.compute(sample, q, s, b).float()
                rel = ((got-ref).norm() / ref.norm().clamp_min(1e-12)).item()
                ok = bool(torch.isfinite(got).all().item()) and rel <= 0.005
                if key is not None:
                    self.checked[id(module)] = (key, weakref.ref(q), weakref.ref(s), ok)
                self.reasons['validation_checks'] += 1
                if not ok:
                    logging.warning("[H3 Packed] validation rejected shape=%s rel=%.6g", tuple(q.shape), rel)
            else:
                self.reasons['validation_cache_hits'] += 1
            if ok:
                y = self.compute(x, q, s, b)
                self.hits += 1
                return y
        except (RuntimeError, ValueError, TypeError, NotImplementedError) as exc:
            self.checked.pop(id(module), None)
            self.reasons['native_exception'] += 1
            logging.warning("[H3 Packed] native fallback: %r", exc)
        self.fallbacks += 1
        self.reasons['validation_or_native_failure'] += 1
        return None

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
