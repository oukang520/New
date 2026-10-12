"""Public Invariant-EvAM interface.

The package exposes identification diagnostics before point estimation.  Small
state-space CTMC internals remain in ``phase2`` for frozen reproducibility.
"""

from .api import (
    diagnose_identifiability,
    fit,
    operator_audit,
    profile_functional,
    sensitivity_analysis,
)

__all__ = [
    "fit",
    "diagnose_identifiability",
    "profile_functional",
    "sensitivity_analysis",
    "operator_audit",
]
