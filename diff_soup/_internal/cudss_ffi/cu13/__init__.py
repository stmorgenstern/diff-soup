"""cuDSS FFI extension built against the CUDA 13 wheels.

Holds the ``_native_solver`` module compiled for CUDA 13 (``nvidia-cudss-cu13``
and the CUDA 13 runtime). The loader in ``diff_soup.linear._cudss._ffi_loader``
imports it when the installed JAX plugin is ``jax_cuda13_plugin``.
"""
