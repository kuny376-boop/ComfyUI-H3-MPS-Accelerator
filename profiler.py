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
    token_histogram: dict[str, int] = field(default_factory=dict)

    def record(
        self,
        *,
        elapsed: float,
        q_tokens: int,
        k_tokens: int,
        masked: bool,
        failed: bool,
    ) -> None:
        self.calls += 1
        self.host_seconds += elapsed
        self.max_tokens = max(self.max_tokens, q_tokens, k_tokens)
        self.masked_calls += int(masked)
        self.cross_calls += int(q_tokens != k_tokens)
        self.failures += int(failed)
        upper = 1
        while upper < max(q_tokens, k_tokens):
            upper <<= 1
        key = f"<={upper}"
        self.token_histogram[key] = self.token_histogram.get(key, 0) + 1

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
            "token_histogram": dict(sorted(self.token_histogram.items())),
        }


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
            with _LOCK:
                stats.record(
                    elapsed=elapsed,
                    q_tokens=q_tokens,
                    k_tokens=k_tokens,
                    masked=mask is not None,
                    failed=failed,
                )

    return override

