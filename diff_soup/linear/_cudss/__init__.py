"""Internal cuDSS wrapper for diff_soup.

``cudss_contract_available`` reports whether cuDSS can serve the analyze /
factor / solve functions in this process. The contract handlers live in
``_contract.py`` and the backend class in ``_backend.py``. The hooks of a
layout that spreads one factorization over the devices of a ``"model"`` mesh
axis (cuDSS MG and MGMN) live in the ``_model_axis`` package: the analysis
record, the contract hooks that call the native per-call solver, and the
multi-process communicator setup.
"""

from diff_soup.linear._cudss._contract import cudss_contract_available

__all__ = ["cudss_contract_available"]
