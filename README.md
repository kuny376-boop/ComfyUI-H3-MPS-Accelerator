# ComfyUI H3 MPS Accelerator

Experimental Apple Silicon acceleration and measurement nodes for MiniMax H3.
The project is workflow-agnostic: it patches a ComfyUI `MODEL` object and does
not modify or vendor ComfyUI or any other custom-node package.

## Current milestone: measurement and safety

The first milestone deliberately does not approximate model output. It adds:

- **H3 MPS Runtime Guard**: rejects unsafe H3/MPS launch combinations and is a
  pass-through on Windows.
- **H3 MPS Attention Profiler**: wraps the active attention backend, including
  SolAttn when it is already installed, and records call geometry and time,
  H3 block index, diffusion sigma, dtype/device and peak MPS allocation.
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


## Native H3 Ref2VA preflight (A0.1)

Before benchmarking an official or BanZhang H3 workflow, inspect the **actual
MODEL ancestry of the active sampler**. A SolAttnPatch node that exists in a
workflow but only feeds a second-stage branch does not accelerate first-stage
sampling. Likewise, our own profiler/quant nodes must be on that same ancestry
to contribute measurements or speedups. The node's presence is not proof of a
runtime hit.

For the 1117 BanZhang macOS workflow, trace `KSampler.model` back through
`BanZhang_SelfLiftMultiStage` and `CSGlideCastCS`. The supplied 1117 Mac
variant has a separate SolAttnPatch branch feeding the second-stage model;
first-stage sampling goes via the unsol-patched dial output. Do not report
SolAttn first-pass gains for that unmodified workflow.

Keep `backend=observe` for initial H3MPSQuantAccelerator runs and
`exact_mps_timing=false` for whole-run profiling. Only enable experimental
Packed INT8 or ConvRot paths after logging eligible/hit/fallback counts,
checking native operator support and confirming LoRA patching behavior. The
PackedLinear path deliberately rejects prepared modules with active
`weight_function` / `bias_function` patch lists; a 263-patch LoRA run may
therefore have many legitimate fallbacks. Never infer a speedup from node
placement alone.

For reproducible benchmarks record the model and text encoder file hashes,
ComfyUI/PyTorch/macOS/Metal versions, first-stage model ancestry, LoRA name
and patch count, image count, actual frames, sigma schedule, peak MPS memory,
swap in/out delta, sampler latency and VAE decode latency. Start by comparing
fixed latent VAE decode variants separately from the costly 4-step sampler.
