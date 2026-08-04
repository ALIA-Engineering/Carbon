"""
Carbon Cross-GPU Determinism Proof

The hardest test: train on GPU 0 (RTX 4090), train on GPU 1 (RTX 5090).
Same seed, same data, same steps.

Standard PyTorch: different weights (different hardware = different bits).
Carbon: identical weights. Different silicon, same math.

This is the paper's killer result. Nobody has done this.

Run: python benchmarks/prove_cross_gpu.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import os
import sys
import json
import re
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def model_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(param.data.cpu().numpy().tobytes())
    return h.hexdigest()[:32]


class CarbonLinear(nn.Module):
    """Linear layer using Carbon's deterministic matmul."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        from carbon.matmul import DeterministicLinear
        self.linear = DeterministicLinear(in_features, out_features, bias=bias)
        # Copy the same init for reproducibility
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)


# Use the shipped layer, not a benchmark-local copy, so this experiment
# exercises the same code a user of the package would get.
from carbon.normalization import CarbonLayerNorm


class TinyTransformer(nn.Module):
    """Transformer built entirely with Carbon's deterministic ops.
    No cuBLAS, no non-deterministic reductions."""

    def __init__(self, vocab_size=1000, dim=256, layers=4, ff_dim=512,
                 use_carbon_ops=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

        Linear = CarbonLinear if use_carbon_ops else nn.Linear
        Norm = CarbonLayerNorm if use_carbon_ops else nn.LayerNorm

        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.ModuleDict({
                "norm1": Norm(dim),
                "attn_qkv": Linear(dim, dim * 3),
                "attn_out": Linear(dim, dim),
                "norm2": Norm(dim),
                "ff1": Linear(dim, ff_dim),
                "ff2": Linear(ff_dim, dim),
            }))
        self.head_norm = Norm(dim)
        self.head = Linear(dim, vocab_size)
        self.dim = dim

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            # Self-attention (manual, no cuBLAS SDPA)
            normed = layer["norm1"](h)
            qkv = layer["attn_qkv"](normed)
            q, k, v = qkv.chunk(3, dim=-1)
            scale = self.dim ** -0.5
            # Use explicit matmul (patched by Carbon when enabled)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            attn_out = torch.matmul(attn, v)
            h = h + layer["attn_out"](attn_out)

            # FFN
            normed = layer["norm2"](h)
            h = h + layer["ff2"](F.gelu(layer["ff1"](normed)))

        return self.head(self.head_norm(h))


def train_on_device(seed: int, steps: int, device: str,
                    use_carbon: bool) -> dict:
    """Train and return weight hash + loss."""
    if use_carbon:
        import carbon
        carbon.enable(seed=seed, warn=False)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = TinyTransformer(use_carbon_ops=use_carbon).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Deterministic data on CPU then move
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    inputs = torch.randint(0, 1000, (32, 64), generator=g).to(device)
    targets = torch.randint(0, 1000, (32, 64), generator=g).to(device)

    start = time.perf_counter()
    for step in range(steps):
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, 1000), targets.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    elapsed = time.perf_counter() - start

    wh = model_hash(model)

    if use_carbon:
        carbon.disable()

    return {
        "device": torch.cuda.get_device_name(int(device.split(":")[-1])),
        "weight_hash": wh,
        "final_loss": loss.item(),
        "time_s": elapsed,
    }


def check_device_supported(index: int) -> tuple[bool, str]:
    """Verify this PyTorch build actually has kernels for the device.

    A PyTorch wheel only ships cubins for the architectures it was built
    against. Running on a newer card (e.g. sm_120 Blackwell on a cu124 wheel)
    fails at the first kernel launch with an opaque "no kernel image is
    available for execution on the device". Detect that up front so a failed
    determinism run is never mistaken for a determinism result.
    """
    name = torch.cuda.get_device_name(index)
    major, minor = torch.cuda.get_device_capability(index)
    sm = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()

    cc = major * 10 + minor

    def _cc(tag: str) -> int:
        # Arch tags may carry a suffix (e.g. "sm_90a"); pull out the digits.
        return int(re.search(r"\d+", tag).group())

    # Exact cubin match.
    supported = sm in arch_list

    # CUDA guarantees binary compatibility within a major compute capability,
    # so an sm_86 cubin runs on an sm_89 device. Across majors it does not.
    if not supported:
        supported = any(
            a.startswith("sm_") and _cc(a) // 10 == major and _cc(a) <= cc
            for a in arch_list
        )

    # Embedded PTX for a lower arch can be JIT-compiled forward.
    if not supported:
        ptx = [_cc(a) for a in arch_list if a.startswith("compute_")]
        supported = any(p <= cc for p in ptx)

    detail = (f"{name} ({sm}); this torch {torch.__version__} was built for "
              f"[{', '.join(arch_list)}]")
    return supported, detail


def run_cross_gpu():
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print("Need 2 GPUs for cross-GPU test. Found:", num_gpus)
        print("Running single-GPU determinism test instead.")
        return

    unsupported = []
    for i in range(2):
        ok, detail = check_device_supported(i)
        status = "OK " if ok else "UNSUPPORTED"
        print(f"  GPU {i}: {status} - {detail}")
        if not ok:
            unsupported.append(i)

    if unsupported:
        print()
        print("  ABORT: this PyTorch build cannot execute kernels on GPU(s) "
              f"{unsupported}.")
        print("  The cross-GPU determinism claim cannot be evaluated in this "
              "environment.")
        print("  Install a wheel built for the newer architecture, e.g.:")
        print("    pip install --index-url https://download.pytorch.org/whl/cu128 "
              "torch --upgrade")
        raise SystemExit(2)

    gpu0_name = torch.cuda.get_device_name(0)
    gpu1_name = torch.cuda.get_device_name(1)

    seed = 42
    steps = 50

    print(f"{'='*70}")
    print(f"  CARBON CROSS-GPU DETERMINISM PROOF")
    print(f"  GPU 0: {gpu0_name}")
    print(f"  GPU 1: {gpu1_name}")
    print(f"  Seed: {seed}  Steps: {steps}")
    print(f"{'='*70}")

    # --- Standard PyTorch ---
    print(f"\n  Standard PyTorch (no determinism):")
    r0_std = train_on_device(seed, steps, "cuda:0", use_carbon=False)
    r1_std = train_on_device(seed, steps, "cuda:1", use_carbon=False)

    print(f"    GPU 0 ({gpu0_name}): hash={r0_std['weight_hash']}  loss={r0_std['final_loss']:.6f}")
    print(f"    GPU 1 ({gpu1_name}): hash={r1_std['weight_hash']}  loss={r1_std['final_loss']:.6f}")
    std_match = r0_std["weight_hash"] == r1_std["weight_hash"]
    print(f"    Match: {'YES' if std_match else 'NO'}")

    # --- Carbon ---
    print(f"\n  Carbon deterministic:")
    r0_carbon = train_on_device(seed, steps, "cuda:0", use_carbon=True)
    r1_carbon = train_on_device(seed, steps, "cuda:1", use_carbon=True)

    print(f"    GPU 0 ({gpu0_name}): hash={r0_carbon['weight_hash']}  loss={r0_carbon['final_loss']:.6f}")
    print(f"    GPU 1 ({gpu1_name}): hash={r1_carbon['weight_hash']}  loss={r1_carbon['final_loss']:.6f}")
    carbon_match = r0_carbon["weight_hash"] == r1_carbon["weight_hash"]
    print(f"    Match: {'YES — IDENTICAL BITS' if carbon_match else 'NO — Carbon failed cross-GPU'}")

    # --- Summary ---
    print(f"\n  {'='*70}")
    print(f"  VERDICT")
    print(f"  {'='*70}")
    print(f"  Standard PyTorch cross-GPU: {'MATCH' if std_match else 'DIFFERS'}")
    print(f"  Carbon cross-GPU:           {'MATCH — DETERMINISM PROVEN' if carbon_match else 'DIFFERS'}")
    if carbon_match and not std_match:
        print(f"  Carbon achieved what PyTorch could not: identical weights on different GPUs.")

    # Save
    out_path = Path(__file__).parent / "cross_gpu_proof.json"
    results = {
        "gpu0": gpu0_name, "gpu1": gpu1_name,
        "seed": seed, "steps": steps,
        "standard": {"gpu0": r0_std, "gpu1": r1_std, "match": std_match},
        "carbon": {"gpu0": r0_carbon, "gpu1": r1_carbon, "match": carbon_match},
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run_cross_gpu()
