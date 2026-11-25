"""Experimental rotated-basis W8A16 reference and standalone benchmark.

Not installed as a model patch. Run directly on the target Mac first.
"""
import argparse
import time
import torch
import torch.nn.functional as F


def rotated_linear(x, q, scale, bias=None, group=256):
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation
    if q.ndim != 2 or q.dtype != torch.int8 or x.shape[-1] != q.shape[1]:
        raise ValueError("unsupported weight/input")
    if scale.numel() not in (1, q.shape[0]) or q.shape[1] % group:
        raise ValueError("unsupported scale/group")
    if x.dtype != torch.bfloat16 or x.device != q.device or scale.device != q.device:
        raise ValueError("expected BF16 input and colocated tensors")
    h = _build_hadamard(group, device=x.device, dtype=torch.float32)
    xr = _rotate_activation(x.float(), h, group).to(x.dtype)
    w = (q.float() * scale.float().reshape(-1, 1)).to(x.dtype)
    return F.linear(xr, w, bias)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--in-features", type=int, default=2048)
    p.add_argument("--out-features", type=int, default=6144)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--per-row", action="store_true")
    a = p.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required")
    if min(a.tokens, a.in_features, a.out_features, a.repeats) <= 0 or a.in_features % 256:
        raise ValueError("positive dimensions; input divisible by 256")
    torch.manual_seed(42)
    q = torch.randint(-127, 128, (a.out_features, a.in_features), dtype=torch.int8).to("mps")
    x = torch.randn(a.tokens, a.in_features, dtype=torch.float32).to("mps", dtype=torch.bfloat16)
    scale = torch.tensor(0.0037, device="mps", dtype=torch.float32)
    if a.per_row:
        scale = (0.001 + 0.006 * torch.rand(a.out_features, 1)).to("mps")
    op = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype.default
    def native():
        return F.linear(x, op(q, scale, 256, 2))
    def rotated():
        return rotated_linear(x, q, scale)
    def measure(fn):
        y = fn()
        torch.mps.synchronize()
        del y
        start = time.perf_counter()
        for _ in range(a.repeats):
            y = fn()
            torch.mps.synchronize()
            del y
        return (time.perf_counter() - start) / a.repeats
    with torch.inference_mode():
        ref, got = native(), rotated()
        diff = got.float() - ref.float()
        print("W8A16 max_abs", diff.abs().max().item(),
              "relative_l2", (diff.norm() / ref.float().norm().clamp_min(1e-12)).item())
        del ref, got, diff
        print("native seconds", measure(native))
        print("rotated seconds", measure(rotated))
        from convrot_metal import ConvRotMetal
        metal = ConvRotMetal()
        def fused():
            return F.linear(x, metal.call(op, (q, scale, 256, 2), {}))
        print("metal seconds", measure(fused), "hits", metal.hits,
              "fallbacks", metal.fallbacks, "disabled", metal.disabled)


if __name__ == "__main__":
    import comfy_kitchen
    main()
