"""
Deterministic normalization layers.

`torch.nn.LayerNorm` dispatches into a fused C++/CUDA kernel whose reduction
split depends on the launch configuration, which in turn depends on the SM
count of the device. Two GPUs with different SM counts therefore produce
different low-order bits for the same input, and `carbon.enable()` cannot fix
this by patching `torch.matmul` because the normalization never goes through
that path.

`CarbonLayerNorm` performs the mean/variance reduction in float64 so that the
rounding of the reduction is insensitive to how the sum is split, then casts
back to the input dtype. This is the layer used in the cross-GPU experiments
under `benchmarks/`.

What this does NOT do: it does not make the reduction order canonical, it
relies on float64 having enough headroom that any grouping rounds to the same
float64 value before the final cast. For the tensor widths in the benchmarks
(dim <= 4096, fp32 inputs) that holds; for very wide layers, very large
magnitudes, or float64 inputs it is not guaranteed. Use
`carbon.CompensatedSum` if you need an order-independent reduction with a
proven error bound, at substantially higher cost.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CarbonLayerNorm(nn.Module):
    """LayerNorm with a float64 reduction, for cross-GPU bit-exactness.

    Args:
        normalized_shape: size of the trailing dimension to normalize over.
        eps: added to the variance before the square root.
        elementwise_affine: if True, learn per-element weight and bias.
        accumulate_dtype: dtype used for the mean/variance reduction.

    Shape:
        Input ``(..., normalized_shape)``, output the same.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 accumulate_dtype: torch.dtype = torch.float64):
        super().__init__()
        if isinstance(normalized_shape, (tuple, list)):
            if len(normalized_shape) != 1:
                raise ValueError(
                    "CarbonLayerNorm normalizes over a single trailing "
                    f"dimension; got normalized_shape={normalized_shape}."
                )
            normalized_shape = normalized_shape[0]
        self.normalized_shape = int(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.accumulate_dtype = accumulate_dtype

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.normalized_shape:
            raise ValueError(
                f"CarbonLayerNorm expected last dimension "
                f"{self.normalized_shape}, got {x.shape[-1]}."
            )
        xf = x.to(self.accumulate_dtype)
        mean = xf.mean(dim=-1, keepdim=True)
        centered = xf - mean
        var = (centered * centered).mean(dim=-1, keepdim=True)
        normed = (centered / torch.sqrt(var + self.eps)).to(x.dtype)
        if self.elementwise_affine:
            return normed * self.weight + self.bias
        return normed

    def extra_repr(self) -> str:
        return (f"{self.normalized_shape}, eps={self.eps}, "
                f"elementwise_affine={self.elementwise_affine}, "
                f"accumulate_dtype={self.accumulate_dtype}")
