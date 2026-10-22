"""BitBlame — dynamic quantization forensics for the llama.cpp/GGUF ecosystem.

Implements METHODOLOGY.md (spec v1.1): mock-quantization LOLO sensitivity
screen with directional KL, paired-bootstrap BCa confidence intervals,
token-category blame, exact top-k verification, DP bit-budget optimizer
with the llama.cpp remap-aware size model, imatrix agreement diagnostic,
and a closed-loop validation harness.
"""

__version__ = "0.1.0"
__author__ = "Habib Ur Rehman"
__spec_version__ = "1.1"

from .config import ScanConfig, ProbeConfig  # noqa: F401
