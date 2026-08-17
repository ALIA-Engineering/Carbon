<p align="center">
  <h1 align="center">CARBON</h1>
  <p align="center"><b>Same seed. Different GPU. Identical weights.</b></p>
  <p align="center">
    <a href="https://pypi.org/project/alia-carbon/"><img src="https://img.shields.io/pypi/v/alia-carbon?color=blue&label=PyPI" alt="PyPI"></a>
    <a href="https://github.com/ALIA-Engineering/Carbon/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
    <a href="#the-proof"><img src="https://img.shields.io/badge/4090%20%3D%205090-bit--exact-brightgreen" alt="Deterministic"></a>
  </p>
</p>

---

Carbon makes training bit-exact reproducible across different GPU architectures.

Train on an RTX 4090. Train on an RTX 5090. Same SHA-256 hash on every weight tensor. Same optimizer state. Same loss. Different silicon, identical bits.

Built by [Tushar Sharma](https://github.com/TxsharDev) at [ALIA Labs](https://alialabs.org).

## Install

```bash
pip install alia-carbon
```

## The Problem

```
RTX 4090:  hash = 6a6e2bc1...  loss = 0.069844
RTX 5090:  hash = 46681ef8...  loss = 0.069844
```

Same loss. **Different weights.** cuBLAS picked different internal algorithms on different silicon. Standard PyTorch cannot reproduce this training run on different hardware.

## The Fix

```python
import carbon
carbon.enable(seed=42)
```

```
RTX 4090:  hash = 62118e9c...  loss = 0.070026
RTX 5090:  hash = 62118e9c...  loss = 0.070026
```

**Identical.** 5 seeds tested, up to 500 steps, every configuration matches.

## How It Works

Every non-deterministic op replaced with a deterministic one:

| Op | Standard | Carbon |
|----|----------|--------|
| MatMul | cuBLAS (arch-dependent) | Tiled fp64 + Kahan accumulation |
| LayerNorm | Parallel reduction (thread-dependent) | fp64 mean/variance |
| AllReduce | NCCL (arrival-order-dependent) | AllGather + rank-order reduce |
| Scatter | Atomic race conditions | Sorted index ops |

The mechanism: split every matmul into tiles, upcast to float64, accumulate with [Kahan-Babushka-Neumaier](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) compensation in a fixed order. The float64 computation eliminates architecture-dependent rounding. The fixed order eliminates parallelism-dependent summation differences.

## The Proof

### Toy Model (500K params, 500 steps, 5 seeds)

| Seed | Steps | 4090 Hash | 5090 Hash | Match |
|------|-------|-----------|-----------|-------|
| 42 | 500 | `d6830c89...` | `d6830c89...` | **yes** |
| 123 | 500 | `44843c64...` | `44843c64...` | **yes** |
| 7 | 500 | `7bf2c902...` | `7bf2c902...` | **yes** |
| 999 | 500 | `8cc60024...` | `8cc60024...` | **yes** |
| 2024 | 500 | `1e420d04...` | `1e420d04...` | **yes** |

5 out of 5 configurations. Every hash matches.

### GPT-2 124M Fine-Tune (60M trainable, 20 steps)

| Run | Hash | Loss |
|-----|------|------|
| Standard PyTorch (5090) | `85b72d9f...` | 7.1904 |
| Carbon run 1 (5090) | `995d4c9b...` | 7.1904 |
| Carbon run 2 (5090) | `995d4c9b...` | 7.1904 |
| Carbon cross-GPU (4090) | `995d4c9b...` | 7.1904 |

Three Carbon runs, two GPUs, one hash. Standard PyTorch produces a different hash.

## Overhead

| Scale | Overhead |
|-------|----------|
| 500K toy model | 1.07x |
| GPT-2 124M (60M trainable) | **10.1x** |

The cost of bit-exact determinism. fp64 Kahan-compensated matmul is slower than cuBLAS. For alignment research and debugging where you need exact reproducibility, it's worth it.

## Important: What Carbon Requires

Cross-architecture determinism requires replacing `nn.Linear` with `DeterministicLinear` and `nn.LayerNorm` with `CarbonLayerNorm`. The `carbon.enable()` call patches `torch.matmul` globally, but standard PyTorch modules use internal C++ paths that bypass the patch.

This is not a one-line fix for existing training code. It's a mechanism that works when you build with Carbon's layers.

## Tested On

RTX 4090 (Ada) | RTX 5090 (Blackwell) | H100 SXM | A100 SXM

Consumer GPUs match each other. Datacenter GPUs match each other. Cross-tier (consumer vs datacenter) produces different hashes. Documented, not hidden.

## Citation

```bibtex
@article{sharma2026carbon,
  title={Carbon: Bit-Exact Deterministic Training Across Consumer GPU Architectures},
  author={Sharma, Tushar},
  year={2026},
  url={https://github.com/ALIA-Engineering/Carbon}
}
```

## Roadmap

**v0.1** - Proof of concept. KBN summation, tiled fp64 matmul, Python-level tiling.

**v0.2 (current)** - WebGPU backend. The same bit-exact algorithms (Kahan-compensated tiled matmul, fixed-order reduction, compensated summation) as WGSL compute shaders, no CUDA required. Works on any WebGPU-capable GPU (NVIDIA, AMD, Intel, Apple Silicon). Install with `pip install alia-carbon[wgpu]`.

**v0.3** - Performance. Replace Kahan accumulation with [superaccumulators](https://github.com/radfordneal/xsum) for order-independent exact summation. This eliminates the need for fixed tile order and could close the consumer-vs-datacenter hash gap. CUDA-native tiled matmul to cut the 10x overhead to under 2x.

**v0.4** - Scale. Mixed precision (bf16 forward, fp32 master weights). Multi-GPU deterministic allreduce. FSDP/DDP integration. Target: deterministic fine-tuning of 7B+ models.

## License

Apache-2.0 | ALIA Labs
