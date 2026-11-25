"""Opt-in MPS ConvRot dequantization, scoped by TorchDispatchMode."""
import logging
import torch

_SHADER = r"""
#include <metal_stdlib>
using namespace metal;
kernel void convrot256(const device char* q [[buffer(0)]],
                       const device float* scale [[buffer(1)]],
                       device bfloat* out [[buffer(2)]],
                       uint i [[thread_position_in_grid]],
                       uint t [[thread_position_in_threadgroup]]) {
    threadgroup float a[256];
    // Integer butterfly is exact for 256 signed INT8 values.
    a[t] = float(q[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < 256; stride *= 4) {
        uint digit = (t / stride) % 4;
        uint base = t - digit * stride;
        float v = (a[base] + a[base+stride] +
                   a[base+2*stride] + a[base+3*stride])
                   - 2.0f * a[base+(3-digit)*stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        a[t] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[i] = bfloat((a[t] * 0.0625f) * scale[0]);
}
"""

class ConvRotMetal:
    def __init__(self):
        self.lib = None
        self.disabled = False
        self.checked = set()
        self.hits = 0
        self.fallbacks = 0

    def _compute(self, q, scale):
        out = torch.empty_like(q, dtype=torch.bfloat16)
        self.lib.convrot256(q, scale.reshape(1), out,
                            threads=(q.numel(),), group_size=(256,))
        return out

    def call(self, func, args, kwargs):
        if self.disabled:
            self.fallbacks += 1
            return func(*args, **kwargs)
        if kwargs or len(args) != 4:
            self.fallbacks += 1
            return func(*args, **kwargs)
        q, scale, group, code = args
        if not (isinstance(q, torch.Tensor) and isinstance(scale, torch.Tensor)
                and q.device.type == "mps" and scale.device == q.device
                and q.dtype == torch.int8 and q.ndim == 2
                and q.is_contiguous() and scale.is_contiguous()
                and q.storage_offset() == 0 and scale.storage_offset() == 0
                and scale.dtype == torch.float32 and scale.numel() == 1
                and group == 256 and code == 2 and q.shape[1] % 256 == 0
                and 0 < q.numel() < 2**31):
            self.fallbacks += 1
            return func(*args, **kwargs)
        try:
            if self.lib is None:
                self.lib = torch.mps.compile_shader(_SHADER)
            key = (tuple(q.shape), float(scale.item()))
            if key not in self.checked:
                # Sample beginning, middle and end; keep complete rotation groups.
                n = min(8, q.shape[0])
                starts = sorted({0, (q.shape[0] - n) // 2, q.shape[0] - n})
                sample = torch.cat([q[i:i+n] for i in starts], dim=0).contiguous()
                ref = func(sample, scale, group, code)
                got = self._compute(sample, scale)
                if ref.shape != got.shape or ref.dtype != got.dtype:
                    raise RuntimeError("ConvRot shape/dtype mismatch")
                # CPU float64 diagnostics avoid MPS reduction rounding.
                a = ref.detach().cpu().double()
                b = got.detach().cpu().double()
                if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                    raise RuntimeError("ConvRot nonfinite sample")
                diff = (a - b).abs()
                rel = (diff.norm() / a.norm().clamp_min(1e-30)).item()
                peak = a.abs().max().item()
                max_abs = diff.max().item()
                # BF16 spacing at each reference magnitude. Near cancellation,
                # allow a small FP32 accumulation error relative to sample peak.
                exponent = torch.frexp(a.abs())[1].to(torch.float64)
                ulp = torch.pow(2.0, exponent - 8).clamp_min(2.0 ** -133)
                ulp = torch.where(a == 0, torch.full_like(ulp, 2.0 ** -133), ulp)
                local_ok = bool((diff <= ulp + peak * 2e-6).all())
                passed = rel <= 1e-5 and max_abs <= peak * 1e-4 and local_ok
                logging.info("[H3 ConvRot] sample passed=%s relative_l2=%.9g max_abs=%.9g mismatch=%s/%s local_ok=%s",
                             passed, rel, max_abs, (a != b).sum().item(), a.numel(), local_ok)
                if not passed:
                    raise RuntimeError("ConvRot BF16 tolerance sample mismatch")
                self.checked.add(key)
            out = self._compute(q, scale)
            self.hits += 1
            return out
        except Exception as exc:
            self.disabled = True
            self.fallbacks += 1
            logging.warning("[H3 ConvRot] disabled; native fallback: %r", exc)
            return func(*args, **kwargs)
