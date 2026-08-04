"""
Carbon: Deterministic Deep Learning Training

Achieves bit-exact reproducibility across different GPU counts,
hardware generations, parallelism strategies, and frameworks.
"""

__version__ = "0.2.1"

from carbon.deterministic import enable, disable, is_enabled
from carbon.reduction import DeterministicAllReduce, DeterministicReduceScatter
from carbon.matmul import DeterministicMatMul, DeterministicLinear
from carbon.normalization import CarbonLayerNorm
from carbon.summation import CompensatedSum, KahanAccumulator
from carbon.wrapper import DeterministicTrainer

# WebGPU backend — lazy import to avoid hard dependency on wgpu
def get_wgpu_engine(**kwargs):
    """Get a WebGPU deterministic engine (requires `pip install wgpu`)."""
    from carbon.wgpu_backend import WgpuDeterministicEngine
    return WgpuDeterministicEngine(**kwargs)

__all__ = [
    "enable",
    "disable",
    "is_enabled",
    "DeterministicAllReduce",
    "DeterministicReduceScatter",
    "DeterministicMatMul",
    "DeterministicLinear",
    "CarbonLayerNorm",
    "CompensatedSum",
    "KahanAccumulator",
    "DeterministicTrainer",
    "get_wgpu_engine",
]
