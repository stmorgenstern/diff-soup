#!/usr/bin/env bash
# Build the SuiteSparse 7.10.1 libraries the KLU extension links, from source,
# into a tree laid out like vendor/, for a machine the vendored libraries do not
# cover (they are x86-64 only).
#
# Usage:
#   bash diff_soup/_internal/klu_ffi/build_suitesparse.sh PREFIX
#
# PREFIX receives what CMakeLists.txt reads through KLU_VENDOR_DIR:
#   include/suitesparse/{klu,amd,colamd,btf,SuiteSparse_config}.h
#   lib/libklu.so.2 libamd.so.3 libcolamd.so.3 libbtf.so.2
#       libsuitesparseconfig.so.7   one file per library, named by its soname,
#                                   with RPATH $ORIGIN
#   lib/libgomp.so.1                the C compiler's OpenMP runtime, which
#                                   libsuitesparseconfig needs and the
#                                   SuperLU_MT extension shares
# The build turns Fortran, CUDA and CHOLMOD off, so the libraries need nothing
# else at run time beyond the C library. The libraries are unmodified.
#
# Environment:
#   SUITESPARSE_SOURCE  an unpacked SuiteSparse 7.10.1 tree, or its release
#                       tarball, to build instead of downloading the tarball
#   BUILD_DIR           scratch directory (default: build/suitesparse)
#   JOBS                parallel compile jobs (default: 8)
#   CC                  C compiler (default: gcc)
#
# The tarball is the one conda-forge builds SuiteSparse 7.10.1 from (the
# vendored x86-64 libraries are that build), checked against its SHA-256. The
# build needs CMake >= 3.22, a C compiler with OpenMP, patchelf, and curl unless
# SUITESPARSE_SOURCE is set. KLU and BTF are LGPL-2.1-or-later; AMD, COLAMD and
# SuiteSparse_config are BSD-3-Clause; see THIRD_PARTY_NOTICES.md beside this
# script.
set -euo pipefail

PREFIX="${1:?usage: build_suitesparse.sh PREFIX}"
VERSION=7.10.1
URL="https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/v${VERSION}.tar.gz"
SHA256=9e2974e22dba26a3cffe269731339ae8e01365cfe921b06be6359902bd05862c
BUILD_DIR="${BUILD_DIR:-build/suitesparse}"
JOBS="${JOBS:-8}"
CC="${CC:-gcc}"

mkdir -p "${BUILD_DIR}" "${PREFIX}"
BUILD_DIR="$(cd "${BUILD_DIR}" && pwd)"
PREFIX="$(cd "${PREFIX}" && pwd)"

# The source tree: an unpacked tree as given, else the tarball, checked.
SOURCE="${SUITESPARSE_SOURCE:-}"
if [[ -n "${SOURCE}" && -d "${SOURCE}" ]]; then
    SRC="$(cd "${SOURCE}" && pwd)"
else
    TARBALL="${SOURCE:-${BUILD_DIR}/SuiteSparse-${VERSION}.tar.gz}"
    if [[ ! -f "${TARBALL}" ]]; then
        curl -sSfL -o "${TARBALL}" "${URL}"
    fi
    echo "${SHA256}  ${TARBALL}" | sha256sum --check --quiet
    rm -rf "${BUILD_DIR}/src"
    mkdir -p "${BUILD_DIR}/src"
    tar -xzf "${TARBALL}" -C "${BUILD_DIR}/src" --strip-components=1
    SRC="${BUILD_DIR}/src"
fi

cmake -S "${SRC}" -B "${BUILD_DIR}/cmake" -G "Unix Makefiles" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}/install" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DSUITESPARSE_ENABLE_PROJECTS="suitesparse_config;amd;colamd;btf;klu" \
    -DSUITESPARSE_USE_STRICT=ON \
    -DSUITESPARSE_USE_OPENMP=ON \
    -DSUITESPARSE_USE_FORTRAN=OFF \
    -DSUITESPARSE_USE_CUDA=OFF \
    -DKLU_USE_CHOLMOD=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_STATIC_LIBS=OFF \
    -DSUITESPARSE_DEMOS=OFF \
    -DBUILD_TESTING=OFF
cmake --build "${BUILD_DIR}/cmake" --parallel "${JOBS}"
cmake --install "${BUILD_DIR}/cmake"

# Stage one real file per library under its soname, as stage_vendor.sh does
# for the vendored tree, each finding the others beside it.
rm -rf "${PREFIX}/include/suitesparse" "${PREFIX}/lib"
mkdir -p "${PREFIX}/include/suitesparse" "${PREFIX}/lib"
for header in klu.h amd.h colamd.h btf.h SuiteSparse_config.h; do
    cp "${BUILD_DIR}/install/include/suitesparse/${header}" "${PREFIX}/include/suitesparse/"
done
for stem in klu amd colamd btf suitesparseconfig; do
    real="$(readlink -f "${BUILD_DIR}/install/lib/lib${stem}.so")"
    soname="$(patchelf --print-soname "${real}")"
    cp "${real}" "${PREFIX}/lib/${soname}"
    chmod u+w "${PREFIX}/lib/${soname}"
    patchelf --force-rpath --set-rpath '$ORIGIN' "${PREFIX}/lib/${soname}"
done
gomp="$(readlink -f "$("${CC}" -print-file-name=libgomp.so.1)")"
if [[ ! -f "${gomp}" ]]; then
    echo "build_suitesparse.sh: ${CC} names no libgomp.so.1." >&2
    exit 1
fi
cp "${gomp}" "${PREFIX}/lib/libgomp.so.1"
chmod u+w "${PREFIX}/lib/libgomp.so.1"

echo "SuiteSparse ${VERSION} (KLU, BTF, AMD, COLAMD, SuiteSparse_config) built with" \
    "$("${CC}" --version | head -1), staged into ${PREFIX}:"
ls -l "${PREFIX}/lib"
