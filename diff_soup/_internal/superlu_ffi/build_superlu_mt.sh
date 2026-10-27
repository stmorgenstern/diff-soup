#!/usr/bin/env bash
# Build SuperLU_MT and install it where the diff_soup SuperLU extension finds it.
#
# Usage:
#   bash diff_soup/_internal/superlu_ffi/build_superlu_mt.sh [PREFIX]
#
# PREFIX defaults to $CONDA_PREFIX. The script installs into it:
#   lib/libsuperlu_mt_OPENMP.a    SuperLU_MT, OpenMP threads, 32-bit indices, -fPIC
#   lib/libsuperlu_mt_blas.a      the reference BLAS that ships with SuperLU_MT
#   include/superlu_mt/*.h        the SuperLU_MT headers
# Then `pip install .` (or a CMake build of diff_soup/_internal) builds the
# _native_superlu extension against it; pass
# `-C cmake.define.SUPERLU_MT_ROOT=<PREFIX>` when PREFIX is not the active
# conda prefix.
#
# Environment:
#   SUPERLU_MT_SOURCE  an existing SuperLU_MT checkout to build instead of
#                      cloning (for hosts without network access)
#   SUPERLU_MT_COMMIT  the commit to build (default below)
#   BUILD_DIR          scratch directory (default: build/superlu_mt)
#   JOBS               parallel compile jobs (default: 8)
#   CC                 C compiler (default: gcc)
#
# The build needs git (unless SUPERLU_MT_SOURCE is set), CMake >= 3.13, a C
# compiler with OpenMP, and make. SuperLU_MT is BSD-3 licensed; see
# diff_soup/_internal/superlu_ffi/THIRD_PARTY_NOTICES.md.
set -euo pipefail

PREFIX="${1:-${CONDA_PREFIX:-}}"
if [ -z "$PREFIX" ]; then
    echo "build_superlu_mt.sh: give an install prefix or activate a conda environment." >&2
    exit 2
fi
COMMIT="${SUPERLU_MT_COMMIT:-1e4bc160e900d601f31410f02ff77ecab59ad275}"
BUILD_DIR="${BUILD_DIR:-build/superlu_mt}"
JOBS="${JOBS:-8}"
CC="${CC:-gcc}"

mkdir -p "$BUILD_DIR"
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"

if [ -n "${SUPERLU_MT_SOURCE:-}" ]; then
    SRC="$(cd "$SUPERLU_MT_SOURCE" && pwd)"
else
    SRC="$BUILD_DIR/src"
    if [ ! -d "$SRC/.git" ]; then
        git clone https://github.com/xiaoyeli/superlu_mt "$SRC"
    fi
    git -C "$SRC" checkout --quiet "$COMMIT"
fi

# The reference BLAS installs as libblas.a; refuse to overwrite another
# package's library of that name.
if [ -e "$PREFIX/lib/libblas.a" ]; then
    echo "build_superlu_mt.sh: $PREFIX/lib/libblas.a already exists and would be overwritten;" \
        "install into another prefix." >&2
    exit 2
fi

cmake -S "$SRC" -B "$BUILD_DIR/cmake" -G "Unix Makefiles" \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-O3 -fPIC" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DTHREAD_API=OPENMP \
    -DLONGINT=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -Denable_internal_blaslib=ON \
    -Denable_examples=OFF \
    -Denable_tests=OFF \
    -Denable_fortran=OFF \
    -Denable_doc=OFF
make -C "$BUILD_DIR/cmake" -j"$JOBS" superlu_mt_OPENMP blas
cmake --install "$BUILD_DIR/cmake"

# Give the reference BLAS a name no other package picks up through -lblas.
mv -f "$PREFIX/lib/libblas.a" "$PREFIX/lib/libsuperlu_mt_blas.a"

echo "SuperLU_MT $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo "from $SRC") installed into $PREFIX"
