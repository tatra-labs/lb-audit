"""Budgeted strong-verifier audits against reward hacking in GRPO."""

from .corrections import CORRECTIONS, caught_reward
from .groups import (
    FAIL,
    GENUINE,
    HACK,
    Group,
    advantage_mass,
    advantages,
    hack_advantage_if_unaudited,
    sample_groups,
)
from .policies import POLICIES, audit
from .trainer import Run, train

__version__ = "0.1.0"
__all__ = [
    "CORRECTIONS",
    "FAIL",
    "GENUINE",
    "Group",
    "HACK",
    "POLICIES",
    "Run",
    "advantage_mass",
    "advantages",
    "audit",
    "caught_reward",
    "hack_advantage_if_unaudited",
    "sample_groups",
    "train",
]
