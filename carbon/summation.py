"""
Compensated Summation — the foundation of Carbon.

Standard floating-point summation is non-associative: (a+b)+c != a+(b+c).
This means different parallelism strategies (which change summation order)
produce different results. Carbon enforces deterministic summation order
using compensated algorithms that track and correct rounding errors.

Two implementations:
- KahanAccumulator: Kahan-Babushka-Neumaier summation (2x cost, exact compensation)
- CompensatedSum: Pairwise summation with fixed tree reduction order
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class KahanAccumulator:
    """
    Kahan-Babushka-Neumaier summation.

    Tracks a running compensation term that captures rounding errors, giving
    an O(eps) error bound instead of O(n*eps) for naive summation.

    KBN is *order-dependent*: feeding the same multiset of values in a
    different sequence can produce different bits. It is not an
    order-independent (reproducible) summation in the ReproBLAS sense. What
    makes results reproducible here is that callers impose a canonical order
    before accumulating -- see ``CompensatedSum._kahan_sum``, which sorts along
    the reduction dimension, and the serial tile loop in ``carbon.matmul``.
    The compensation improves accuracy; the fixed order is what provides
    determinism. Both are required.
    """

    def __init__(self, shape: tuple, dtype: torch.dtype = torch.float64,
                 device: Optional[torch.device] = None):
        self.sum = torch.zeros(shape, dtype=dtype, device=device)
        self.compensation = torch.zeros(shape, dtype=dtype, device=device)
        self.dtype = dtype

    def add(self, value: torch.Tensor):
        """Add a value with Kahan compensation."""
        value = value.to(self.dtype)
        t = self.sum + value
        # If sum is bigger, compensation captures lost low-order bits of value
        # If value is bigger, compensation captures lost low-order bits of sum
        mask = self.sum.abs() >= value.abs()
        self.compensation = torch.where(
            mask,
            self.compensation + ((self.sum - t) + value),
            self.compensation + ((value - t) + self.sum),
        )
        self.sum = t

    def result(self) -> torch.Tensor:
        """Get the compensated sum."""
        return self.sum + self.compensation

    def reset(self):
        self.sum.zero_()
        self.compensation.zero_()


class CompensatedSum(nn.Module):
    """
    Deterministic summation via fixed-order pairwise reduction.

    Instead of relying on CUDA's non-deterministic parallel reduction,
    we enforce a canonical binary tree reduction order. This is slower
    but guarantees bit-exact results regardless of thread scheduling.

    The key insight: if we always reduce in the same order, the rounding
    errors are the same, so the result is the same.
    """

    def __init__(self, use_kahan: bool = True, accumulate_dtype: torch.dtype = torch.float64):
        super().__init__()
        self.use_kahan = use_kahan
        self.accumulate_dtype = accumulate_dtype

    def forward(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Deterministic sum along dimension.

        Args:
            tensor: input tensor
            dim: dimension to sum over

        Returns:
            Sum with deterministic rounding behavior
        """
        if self.use_kahan:
            return self._kahan_sum(tensor, dim)
        return self._pairwise_sum(tensor, dim)

    def _kahan_sum(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """Kahan summation along a dimension."""
        # Move target dim to last position
        tensor = tensor.movedim(dim, -1)
        n = tensor.shape[-1]

        # Sort along reduction dimension for canonical order
        # This ensures identical summation order across different parallelisms
        sorted_tensor, _ = tensor.sort(dim=-1)

        result_shape = sorted_tensor.shape[:-1]
        acc = KahanAccumulator(
            result_shape,
            dtype=self.accumulate_dtype,
            device=tensor.device,
        )

        for i in range(n):
            acc.add(sorted_tensor[..., i])

        return acc.result().to(tensor.dtype)

    def _pairwise_sum(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Pairwise (cascade) summation with fixed tree structure.

        Recursively splits the array in half and sums pairs.
        The tree structure is fixed regardless of array size
        (we pad to next power of 2).
        """
        original_dtype = tensor.dtype
        tensor = tensor.movedim(dim, -1).to(self.accumulate_dtype)
        n = tensor.shape[-1]

        # Pad to next power of 2
        next_pow2 = 1
        while next_pow2 < n:
            next_pow2 *= 2

        if next_pow2 > n:
            padding = torch.zeros(
                *tensor.shape[:-1], next_pow2 - n,
                dtype=self.accumulate_dtype, device=tensor.device
            )
            tensor = torch.cat([tensor, padding], dim=-1)

        # Binary tree reduction
        current = tensor
        while current.shape[-1] > 1:
            half = current.shape[-1] // 2
            left = current[..., :half]
            right = current[..., half:]
            current = left + right

        return current.squeeze(-1).to(original_dtype)


def deterministic_sum(tensor: torch.Tensor, dim: int = -1,
                      method: str = "kahan") -> torch.Tensor:
    """Functional API for deterministic summation."""
    summer = CompensatedSum(use_kahan=(method == "kahan"))
    return summer(tensor, dim)
