# ComfyUI H3 MPS Accelerator

Experimental Apple Silicon acceleration and measurement nodes for MiniMax H3.
This project is an optional companion to ComfyUI-Banzhang-All; it does not
modify or vendor the Banzhang node pack.

## Current milestone: measurement and safety

The first milestone deliberately does not approximate model output. It adds:

- **H3 MPS Runtime Guard**: rejects unsafe H3/MPS launch combinations and is a
  pass-through on Windows.
- **H3 MPS Attention Profiler**: wraps the active attention backend, including
  SolAttn when it is already installed, and records call geometry and time.
- **H3 MPS Profiler Report**: runs after a connected sampler output and prints
  JSON suitable for benchmark comparisons.

`exact_mps_timing=true` synchronizes before and after every attention call. It
is accurate but intentionally slow. Leave it off for normal generation.

## Installation

Place this directory beside the other ComfyUI custom nodes:

```text
ComfyUI/custom_nodes/ComfyUI-H3-MPS-Accelerator
```

For MiniMax H3 on Apple Silicon start ComfyUI with:

```bash
./venv/bin/python main.py --use-pytorch-cross-attention
```

Do not force the H3 model or text encoder to FP16.

## Suggested workflow order

```text
Model loader -> LoRA -> SolAttnPatch -> H3 MPS Runtime Guard
             -> H3 MPS Attention Profiler -> sampler

sampler latent -> H3 MPS Profiler Report
profile output --------------------------^
```

## Roadmap

1. Establish M4 Pro cold/warm baselines.
2. Add block and denoising-step attribution without altering output.
3. Prototype H3-aware spatial/temporal routing on top of the safe fallback.
4. Evaluate fused Metal projection/norm kernels.
5. Add a separate conservative cache profile for 20-step H3 only.

Every approximate path must retain an exact fallback and pass finite-latent,
audio/video decode, quality, and repeat-run memory tests.

## License

Apache-2.0. External projects used for research retain their own licenses; no
SolAttn source code is vendored here.

