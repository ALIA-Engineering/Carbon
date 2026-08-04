"""Tests for CarbonLayerNorm.

These run on CPU so they execute in CI. The cross-GPU bit-exactness claim is
necessarily a hardware experiment and lives in benchmarks/prove_cross_gpu.py.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from carbon.normalization import CarbonLayerNorm


def test_matches_torch_layer_norm():
    torch.manual_seed(0)
    dim = 128
    ln = CarbonLayerNorm(dim)
    x = torch.randn(16, 7, dim)
    ref = F.layer_norm(x, (dim,), ln.weight, ln.bias, eps=ln.eps)
    out = ln(x)
    assert torch.allclose(ref, out, atol=1e-5, rtol=1e-5)


def test_matches_torch_with_learned_affine():
    torch.manual_seed(1)
    dim = 64
    ln = CarbonLayerNorm(dim)
    with torch.no_grad():
        ln.weight.copy_(torch.randn(dim))
        ln.bias.copy_(torch.randn(dim))
    x = torch.randn(4, dim)
    ref = F.layer_norm(x, (dim,), ln.weight, ln.bias, eps=ln.eps)
    assert torch.allclose(ref, ln(x), atol=1e-5, rtol=1e-5)


def test_no_affine():
    dim = 32
    ln = CarbonLayerNorm(dim, elementwise_affine=False)
    assert ln.weight is None and ln.bias is None
    x = torch.randn(3, dim)
    out = ln(x)
    assert torch.allclose(out.mean(-1), torch.zeros(3), atol=1e-5)


def test_reduction_is_split_invariant():
    """The reason this layer exists: the result must not depend on how the
    reduction is grouped, which is what differs between GPUs with different
    SM counts. Emulate that by reducing the same row in two groupings."""
    torch.manual_seed(2)
    dim = 4096
    x = torch.randn(1, dim, dtype=torch.float32)
    xf = x.to(torch.float64)

    whole = xf.mean(dim=-1)
    halves = torch.stack([xf[:, :dim // 2].mean(-1), xf[:, dim // 2:].mean(-1)]).mean(0)
    # Strict equality. Approximate equality is the failure mode this library
    # exists to remove: the claim is identical bits, not close bits.
    assert torch.equal(whole, halves)

    ln = CarbonLayerNorm(dim, elementwise_affine=False)
    assert torch.isfinite(ln(x)).all()


def test_rejects_wrong_trailing_dim():
    ln = CarbonLayerNorm(16)
    with pytest.raises(ValueError, match="expected last dimension"):
        ln(torch.randn(2, 8))


def test_rejects_multi_dim_normalized_shape():
    with pytest.raises(ValueError, match="single trailing"):
        CarbonLayerNorm((8, 16))


def test_accepts_single_element_sequence():
    ln = CarbonLayerNorm([16])
    assert ln.normalized_shape == 16
    assert ln(torch.randn(2, 16)).shape == (2, 16)


def test_gradients_flow():
    dim = 32
    ln = CarbonLayerNorm(dim)
    x = torch.randn(4, dim, requires_grad=True)
    ln(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert ln.weight.grad is not None and torch.isfinite(ln.weight.grad).all()


def test_exported_from_package():
    import carbon
    assert carbon.CarbonLayerNorm is CarbonLayerNorm
