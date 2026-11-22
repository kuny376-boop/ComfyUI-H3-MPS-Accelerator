"""Runtime validation for MiniMax H3 on Apple Silicon."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


UNSAFE_FP16_FLAGS = ("--force-fp16", "--fp16-unet", "--fp16-text-enc")


@dataclass(frozen=True)
class GuardResult:
    platform: str
    device: str
    is_h3: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        state = "OK" if self.ok else "BLOCKED"
        lines = [
            f"H3 MPS Guard: {state}",
            f"platform={self.platform} device={self.device} h3={self.is_h3}",
        ]
        lines.extend(f"ERROR: {item}" for item in self.errors)
        lines.extend(f"WARN: {item}" for item in self.warnings)
        return "\n".join(lines)


def validate_runtime(
    *,
    platform: str,
    device: str,
    is_h3: bool,
    pytorch_attention: bool,
    enabled_flags: Iterable[str] = (),
) -> GuardResult:
    errors: list[str] = []
    warnings: list[str] = []
    flags = set(enabled_flags)

    if platform != "darwin":
        warnings.append("not macOS; accelerator remains a pass-through")
    elif device != "mps":
        warnings.append("macOS detected but ComfyUI is not using MPS")

    if is_h3 and device == "mps":
        if not pytorch_attention:
            errors.append("restart ComfyUI with --use-pytorch-cross-attention")
        unsafe = sorted(flags.intersection(UNSAFE_FP16_FLAGS))
        if unsafe:
            errors.append(
                "remove unsafe forced-FP16 flags: " + ", ".join(unsafe)
            )

    if not is_h3:
        warnings.append("loaded model does not look like MiniMax H3")

    return GuardResult(
        platform=platform,
        device=device,
        is_h3=is_h3,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )

