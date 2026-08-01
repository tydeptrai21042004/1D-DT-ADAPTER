"""Deprecated compatibility shim for releases before v0.7.0.

The canonical implementation is :mod:`models.dt1d_adapter` and the canonical
method name is ``DT1D-Adapter``. New code must import ``DT1DAdapter`` from the
canonical module. This shim is retained only so older checkpoints and scripts
can be migrated without silently breaking.
"""

try:
    from .dt1d_adapter import DT1DAdapter
except ImportError:  # direct file loading in legacy tests/scripts
    from models.dt1d_adapter import DT1DAdapter

HCCAdapter = DT1DAdapter
H1D_DT_Adapter = DT1DAdapter
OneDDTAdapter = DT1DAdapter

__all__ = ["DT1DAdapter", "HCCAdapter", "H1D_DT_Adapter", "OneDDTAdapter"]
