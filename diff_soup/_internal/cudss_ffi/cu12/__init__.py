"""cuDSS FFI extension built against the CUDA 12 wheels.

Holds the ``_native_solver`` module compiled for CUDA 12 (``nvidia-cudss-cu12``
and the CUDA 12 runtime). The loader in ``diff_soup.linear._cudss._ffi_loader``
imports it when the installed JAX plugin is ``jax_cuda12_plugin``.
"""
