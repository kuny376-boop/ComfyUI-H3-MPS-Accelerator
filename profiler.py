"""Thread-safe attention profiler used by the ComfyUI nodes.

GPU work is asynchronous.  Exact MPS timings therefore synchronize around
each attention call and are deliberately a diagnostic mode, not a speed mode.
The default low-overhead mode measures host dispatch time and call geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import threading
import time
from typing import Callable


@dataclass
class ProfileStats:
    name: str
    calls: int = 0
    host_seconds: float = 0.0
    max_tokens: int = 0
    masked_calls: int = 0
    cross_calls: int = 0
    failures: int = 0
    peak_allocated_bytes: int = 0
    token_histogram: dict[str, int] = field(default_factory=dict)
    calls_by_block: dict[str, int] = field(default_factory=dict)
    calls_by_sigma: dict[str, int] = field(default_factory=dict)
    calls_by_device: dict[str, int] = field(default_factory=dict)
    calls_by_dtype: dict[str, int] = field(default_factory=dict)

    def record(
        self,
        *,
        elapsed: float,
        q_tokens: int,
        k_tokens: int,
        masked: bool,
        failed: bool,
        block: int | None = None,
        sigma: float | None = None,
        device: str = "unknown",
        dtype: str = "unknown",
        allocated_bytes: int = 0,
    ) -> None:
        self.calls += 1
        self.host_seconds += elapsed
        self.max_tokens = max(self.max_tokens, q_tokens, k_tokens)
        self.masked_calls += int(masked)
        self.cross_calls += int(q_tokens != k_tokens)
        self.failures += int(failed)
        self.peak_allocated_bytes = max(self.peak_allocated_bytes, allocated_bytes)
        upper = 1
        while upper < max(q_tokens, k_tokens):
            upper <<= 1
        key = f"<={upper}"
        self.token_histogram[key] = self.token_histogram.get(key, 0) + 1
        block_key = "unknown" if block is None else str(block)
        self.calls_by_block[block_key] = self.calls_by_block.get(block_key, 0) + 1
        if sigma is not None:
            sigma_key = f"{sigma:.4g}"
            self.calls_by_sigma[sigma_key] = self.calls_by_sigma.get(sigma_key, 0) + 1
        self.calls_by_device[device] = self.calls_by_device.get(device, 0) + 1
        self.calls_by_dtype[dtype] = self.calls_by_dtype.get(dtype, 0) + 1

    def snapshot(self) -> dict:
        return {
            "session": self.name,
            "attention_calls": self.calls,
            "host_or_synchronized_seconds": round(self.host_seconds, 6),
            "average_ms": round(
                1000.0 * self.host_seconds / self.calls if self.calls else 0.0, 4
            ),
            "max_tokens": self.max_tokens,
            "masked_calls": self.masked_calls,
            "cross_attention_calls": self.cross_calls,
            "failures": self.failures,
            "peak_mps_allocated_mb": round(self.peak_allocated_bytes / 2**20, 2),
            "token_histogram": dict(sorted(self.token_histogram.items())),
            "calls_by_block": _sort_numeric_keys(self.calls_by_block),
            "calls_by_sigma": dict(self.calls_by_sigma),
            "calls_by_device": dict(sorted(self.calls_by_device.items())),
            "calls_by_dtype": dict(sorted(self.calls_by_dtype.items())),
        }


def _sort_numeric_keys(values: dict[str, int]) -> dict[str, int]:
    def key(item):
        raw = item[0]
        return (1, raw) if raw == "unknown" else (0, int(raw))

    return dict(sorted(values.items(), key=key))


def _extract_context(kwargs: dict) -> tuple[int | None, float | None]:
    options = kwargs.get("transformer_options") or {}
    block = options.get("sol_block")
    sigmas = options.get("sigmas")
    sigma = None
    if sigmas is not None:
        try:
            sigma = float(sigmas[0])
        except (IndexError, TypeError, ValueError):
            sigma = None
    try:
        block = int(block) if block is not None else None
    except (TypeError, ValueError):
        block = None
    return block, sigma


_LOCK = threading.RLock()
_SESSIONS: dict[str, ProfileStats] = {}


def reset_session(name: str) -> ProfileStats:
    with _LOCK:
        stats = ProfileStats(name=name)
        _SESSIONS[name] = stats
        return stats


def get_session(name: str) -> ProfileStats:
    with _LOCK:
        return _SESSIONS.setdefault(name, ProfileStats(name=name))


def report_session(name: str) -> str:
    with _LOCK:
        return json.dumps(get_session(name).snapshot(), ensure_ascii=False, indent=2)


def make_profile_override(
    *,
    session: str,
    previous: Callable | None,
    synchronize: Callable[[], None] | None = None,
    memory_probe: Callable[[], int] | None = None,
) -> Callable:
    """Wrap the active attention path without changing its numerical output."""

    stats = reset_session(session)

    def override(
        func,
        q,
        k,
        v,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        **kwargs,
    ):
        common = dict(
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )
        target = func if previous is None else lambda *a, **kw: previous(func, *a, **kw)
        if synchronize is not None:
            synchronize()
        started = time.perf_counter()
        failed = False
        try:
            return target(q, k, v, heads, **common)
        except Exception:
            failed = True
            raise
        finally:
            if synchronize is not None:
                synchronize()
            elapsed = time.perf_counter() - started
            q_tokens = int(q.shape[2] if skip_reshape else q.shape[1])
            k_tokens = int(k.shape[2] if skip_reshape else k.shape[1])
            block, sigma = _extract_context(kwargs)
            device = str(getattr(q, "device", "unknown"))
            dtype = str(getattr(q, "dtype", "unknown"))
            allocated = 0
            if memory_probe is not None:
                try:
                    allocated = int(memory_probe())
                except Exception:
                    allocated = 0
            with _LOCK:
                stats.record(
                    elapsed=elapsed,
                    q_tokens=q_tokens,
                    k_tokens=k_tokens,
                    masked=mask is not None,
                    failed=failed,
                    block=block,
                    sigma=sigma,
                    device=device,
                    dtype=dtype,
                    allocated_bytes=allocated,
                )

    return override

