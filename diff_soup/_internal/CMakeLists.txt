cmake_minimum_required(VERSION 3.18)
project(diff_soup LANGUAGES CXX)

# CMake shim consumed by scikit-build-core, which pyproject.toml points here
# with cmake.source-dir. Build logic for each native extension lives in its
# own sub-directory CMakeLists.txt; this file delegates to each so a single
# `pip install` builds them together:
#
#   cudss_ffi    -> _native_solver   (cuDSS, GPU direct LA)
#   klu_ffi      -> _native_klu      (SuiteSparse KLU, CPU sparse LU)
#   superlu_ffi  -> _native_superlu  (SuperLU_MT, CPU sparse LU)
#
# Each subdir's install() rule places its built *.so into the matching package
# directory inside the wheel so it is importable at runtime. The klu_ffi subdir
# additionally bundles its vendored SuiteSparse and GCC runtime libraries into
# its lib/ so the wheel is self-contained with no conda/system SuiteSparse
# required.

# The cuDSS extension is optional. The default build includes it (GPU
# production wheel); a CPU-only build turns it OFF so no _native_solver.so is
# produced, no NVIDIA build toolchain is required, and the package degrades to
# the KLU / Dense backends via the guarded imports in
# diff_soup.linear. Toggle from pip:
#   pip install . -C cmake.define.DIFF_SOUP_BUILD_CUDSS=OFF
option(DIFF_SOUP_BUILD_CUDSS "Build the cuDSS GPU linear-solver FFI extension" ON)

if(DIFF_SOUP_BUILD_CUDSS)
  add_subdirectory(cudss_ffi)
else()
  message(STATUS "diff_soup: DIFF_SOUP_BUILD_CUDSS=OFF — skipping cuDSS extension (CPU-only build)")
endif()

# The KLU extension links only the vendored SuiteSparse libraries (no NVIDIA
# toolchain). DIFF_SOUP_BUILD_KLU is ON, OFF or AUTO; the vendored tree makes
# the extension always buildable, so AUTO builds it too. Toggle from pip:
#   pip install . -C cmake.define.DIFF_SOUP_BUILD_KLU=OFF
set(DIFF_SOUP_BUILD_KLU "ON" CACHE STRING
    "Build the SuiteSparse KLU CPU linear-solver FFI extension: AUTO, ON or OFF")
if(DIFF_SOUP_BUILD_KLU STREQUAL "AUTO" OR DIFF_SOUP_BUILD_KLU)
  add_subdirectory(klu_ffi)
else()
  message(STATUS "diff_soup: DIFF_SOUP_BUILD_KLU=OFF — skipping the KLU extension")
endif()

# The SuperLU_MT extension (_native_superlu, the threaded sparse LU on the
# host) links a SuperLU_MT installation: lib/libsuperlu_mt_OPENMP.a,
# lib/libsuperlu_mt_blas.a and include/superlu_mt/ under SUPERLU_MT_ROOT
# (default: the active conda prefix).
# diff_soup/_internal/superlu_ffi/build_superlu_mt.sh builds and installs one.
# AUTO (default) builds the extension when that installation is present and
# skips it otherwise; ON requires it; OFF skips it. Toggle from pip:
#   pip install . -C cmake.define.DIFF_SOUP_BUILD_SUPERLU=ON -C cmake.define.SUPERLU_MT_ROOT=<prefix>
set(DIFF_SOUP_BUILD_SUPERLU "AUTO" CACHE STRING
    "Build the SuperLU_MT CPU linear-solver FFI extension: AUTO, ON or OFF")
set(SUPERLU_MT_ROOT "$ENV{CONDA_PREFIX}" CACHE PATH
    "Prefix holding include/superlu_mt and lib/libsuperlu_mt_OPENMP.a")
if(DIFF_SOUP_BUILD_SUPERLU STREQUAL "AUTO")
  if(EXISTS "${SUPERLU_MT_ROOT}/lib/libsuperlu_mt_OPENMP.a"
     AND EXISTS "${SUPERLU_MT_ROOT}/include/superlu_mt/slu_mt_ddefs.h")
    add_subdirectory(superlu_ffi)
  else()
    message(STATUS "diff_soup: no SuperLU_MT under SUPERLU_MT_ROOT='${SUPERLU_MT_ROOT}' — "
                   "skipping the SuperLU extension (diff_soup/_internal/superlu_ffi/build_superlu_mt.sh installs one)")
  endif()
elseif(DIFF_SOUP_BUILD_SUPERLU)
  add_subdirectory(superlu_ffi)
else()
  message(STATUS "diff_soup: DIFF_SOUP_BUILD_SUPERLU=OFF — skipping the SuperLU extension")
endif()
