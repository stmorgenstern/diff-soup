# Installation

```bash
pip install diff-soup
```

For NVIDIA GPUs, install the extra that matches your driver:

```bash
pip install "diff-soup[cu13]"   # CUDA 13, driver 580 or newer
pip install "diff-soup[cu12]"   # CUDA 12, driver 525 or newer
```

Requirements:

- Linux x86_64
- Python 3.12 or 3.13
- glibc 2.28 or newer
- for the GPU extras, an NVIDIA GPU of compute capability 8.0 or newer

The wheel bundles the KLU sparse solver and installs JAX 0.11. It does
not include SuperLU_MT, which a [source build](#install-from-source)
adds. The GPU extras add the CUDA build of JAX and the cuDSS sparse
solver, so no compiler or CUDA toolkit is needed. The `mesh` extra installs
gmsh, which meshes complex geometries, and meshio, which reads and writes mesh
files, for example `pip install "diff-soup[cu13,mesh]"`. OpenCV, which
`mask_to_mesh` uses to trace image masks, installs with diff-soup. gmsh needs
the system OpenGL utility library (`libglu1-mesa` on Debian and Ubuntu,
`mesa-libGLU` on RHEL and Fedora).

## Enable float64

diff-soup computes in double precision by default, so turn on JAX's
64-bit mode at the top of every script:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

Exporting `JAX_ENABLE_X64=1` before starting Python does the same.

`import diff_soup` also turns on JAX's constant hoisting, so a compiled
program takes the mesh arrays it uses as arguments instead of embedding them.
This works whether `jax` is imported before or after `diff_soup`.

## Install from source

A source build needs a C++17 compiler. CMake and Ninja are fetched by the
build when they are not installed.

For the CPU, one command builds the KLU solver:

```bash
git clone https://github.com/stmorgenstern/diff-soup
cd diff-soup
pip install . -C cmake.define.DIFF_SOUP_BUILD_CUDSS=OFF
```

For the cuDSS GPU solver, the build needs that CUDA line's wheels and its
`nvcc` in the environment first, so it runs in two passes:

```bash
pip install ".[cu13]" -C cmake.define.DIFF_SOUP_BUILD_CUDSS=OFF
pip install scikit-build-core nanobind patchelf cmake ninja nvidia-cuda-nvcc
pip install --no-build-isolation --no-deps --force-reinstall .
```

For CUDA 12, use the `cu12` extra, leave out `nvidia-cuda-nvcc`, and point the
last command at a CUDA 12 toolkit's compiler with
`-C cmake.define.DIFF_SOUP_NVCC=/usr/local/cuda-12/bin/nvcc`.

`pip install -e` in the last command installs in development mode, and the
`dev` extra adds the test and lint tools. The SuperLU_MT solver is built when
a SuperLU_MT installation is found; `diff_soup/_internal/superlu_ffi/build_superlu_mt.sh`
builds one.

## Troubleshooting

- Install exactly one of `cu12` and `cu13`. With both JAX CUDA plugins
  installed, the cuDSS solver refuses to load.
- If `libcudss.so.1` is not found, add the `nvidia/*/lib` directories of
  the installed wheels to `LD_LIBRARY_PATH` before starting Python.
- To run on the CPU of a GPU machine, set `JAX_PLATFORMS=cpu`.
