# Changelog

## v0.2.1 — 2026-08-04

Documentation and correctness pass.

- README: seed/step counts corrected to match the proof tables; roadmap updated (v0.2 is current, WebGPU shipped).
- Paper (`paper/carbon.tex`): datacenter GPU runs (H100 SXM, A100 SXM, on RunPod) now documented, with evidence status stated plainly.
- LICENSE: full Apache-2.0 text.
- `prove_cross_gpu.py`: SM-tag parsing handles suffixed arch tags like `sm_90a`.
- `test_normalization.py`: strict equality assertion (the library's contract).
- `carbon/__init__.py`: docstring de-inflated.

## v0.2.0 — 2026-06-30

WebGPU backend. Determinism beyond CUDA.

### Added
- **WebGPU backend** (`carbon.wgpu_backend`): same bit-exact algorithms, no CUDA required
  - `WgpuDeterministicEngine` class with `matmul()`, `sum()`, `reduce()` methods
  - WGSL compute shaders: Kahan-compensated tiled matmul, fixed-order reduction, compensated summation
  - Works on any GPU with WebGPU support (NVIDIA, AMD, Intel, Apple Silicon)
  - Single thread per output element — slow but bit-exact, same tradeoff as the CUDA path
- **`get_wgpu_engine()`** convenience function in top-level `carbon` module
- **Optional dependency**: `pip install alia-carbon[wgpu]` for WebGPU support
- **22 new tests** for the WebGPU backend (auto-skipped if wgpu not installed)

### Changed
- Version bump to 0.2.0
- `pyproject.toml` now has `[wgpu]` and `[all]` optional dependency groups

### Unchanged
- All v0.1 PyTorch/CUDA APIs remain identical: `carbon.enable()`, `DeterministicMatMul`, `DeterministicTrainer`, etc.

---

## v0.1.0 — 2026-06-10

First release. Same bits, any GPU.

### What's in it
- **Kahan compensated summation**: O(eps) error instead of O(n*eps), canonical sorted order
- **Deterministic tiled matmul**: fixed reduction order, Kahan accumulation, original-matmul recursion guard
- **Deterministic allreduce**: AllGather + local reduce in rank order (for multi-GPU)
- **Global enable/disable**: `carbon.enable()` patches torch.matmul, torch.mm, torch.bmm, scatter ops
- **DeterministicLinear**: drop-in nn.Linear replacement
- **DeterministicTrainer**: training wrapper with checkpoint hashing + verification

### Proven
- RTX 4090 vs RTX 5090: **bit-exact identical weights** after 50 and 200 training steps
- Optimizer state (AdamW moments): also bit-exact across GPUs
- Standard PyTorch on same test: different weights every time across GPUs

### Known gaps
- Overhead ~1.07x single-GPU same-architecture (PyTorch deterministic mode handles most of it)
- Cross-GPU determinism uses tiled matmul: overhead depends on model size (measured ~1x on toy model, expect ~2-3x at scale)
- Cross-GPU overhead not yet measured at scale
- No fp16/bf16 mixed precision support yet
- Attention uses manual QKV matmul (no SDPA) for cross-GPU determinism
- Multi-node distributed training not tested yet
