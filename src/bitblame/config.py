"""Configuration dataclasses for the BitBlame pipeline (METHODOLOGY.md spec v1.0)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScanConfig:
    """Stage 0/1 LOLO scan configuration (§2.2, §2.3)."""

    bits_ref: int = 4          # b_ref for mock-noise screen
    bits_null: int = 8         # null-floor run bit-width
    group_size: int = 256      # quantization group (k-quant super-block)
    n_draws: int = 2           # R independent noise draws per unit
    n_null_draws: int = 1      # draws for the null-floor run
    n_seqs: int = 32           # N calibration sequences (split-half gate may grow)
    seq_len: int = 512         # tokens per calibration sequence (§12: 512 for 135M)
    batch_size: int = 2        # sequences per forward batch (CPU-memory bound)
    seed: int = 20261014       # master seed
    split_half_rho: float = 0.9  # reliability gate threshold (§2.3)
    max_n_seqs: int = 256      # ceiling when doubling N
    granularity: str = "block"  # "block" (default) or "fine" (block, role)


@dataclass
class ProbeConfig:
    """Stage 3/5 behavioral probe battery configuration (§5.2)."""

    n_mmlu: int = 100
    n_gsm8k: int = 60
    n_humaneval: int = 30
    n_needle: int = 40
    n_copy: int = 20
    max_new_tokens: int = 128
    humaneval_timeout_s: int = 5
    seed: int = 20261014


@dataclass
class BootstrapConfig:
    """Paired-bootstrap configuration (§6)."""

    b_units: int = 2000        # replicates for per-unit / category CIs
    b_probes: int = 10000      # replicates for paired probe gain
    level: float = 0.95
    seed: int = 20261014


@dataclass
class OptimizerConfig:
    """Bit-budget optimizer configuration (§7)."""

    menu_bits: tuple = (2, 3, 4, 5, 6, 8)
    gguf_types: dict = field(default_factory=lambda: {
        2: "Q2_K", 3: "Q3_K", 4: "Q4_K", 5: "Q5_K", 6: "Q6_K", 8: "Q8_0"})
    default_bits: int = 4
    overhead: float = 1.02     # scale/metadata overhead
    alpha_theory: float = 1.3862943611198906  # 2*ln(2), §7.2
