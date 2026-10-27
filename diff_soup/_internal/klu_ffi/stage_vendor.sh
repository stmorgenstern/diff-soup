#!/usr/bin/env bash
# Stage the vendored SuiteSparse KLU headers and prebuilt runtime libraries
# into ``vendor/`` so the wheel build is fully SELF-CONTAINED with NO conda /
# system SuiteSparse and NO build-time network access.
#
# Run ONCE by a developer (against the pinned conda-forge SuiteSparse 7.10.1
# stack in the diffsoup-test env); its output (vendor/include + vendor/lib) is
# COMMITTED to the repo.  At user `pip install` time CMake just reads from
# vendor/ -- it never touches conda and never downloads anything.
#
# Re-stage (e.g. after a SuiteSparse bump):
#   module load miniforge && conda activate diffsoup-test && unset LD_LIBRARY_PATH
#   bash diff_soup/_internal/klu_ffi/stage_vendor.sh
#
# The set of bundled .so's is the full transitive NEEDED tree of _native_klu:
#   direct:      libklu.so.2
#   transitive:  libamd.so.3  libcolamd.so.3  libbtf.so.2  libsuitesparseconfig.so.7
#   GCC runtime: libgfortran.so.5  libquadmath.so.0  libgomp.so.1  libgcc_s.so.1
# Every conda-forge lib already carries RPATH=$ORIGIN/, so once co-located they
# resolve each other with no patching.
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: CONDA_PREFIX unset. Activate diffsoup-test first." >&2
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="${HERE}/vendor"
SRC_INC="${CONDA_PREFIX}/include"
SRC_LIB="${CONDA_PREFIX}/lib"

echo "Staging vendored SuiteSparse from ${CONDA_PREFIX} -> ${VENDOR}"
rm -rf "${VENDOR}"
mkdir -p "${VENDOR}/include/suitesparse" "${VENDOR}/lib"

# --- headers (build-time only; excluded from the wheel) --------------------
# The SuiteSparse headers that klu.h transitively pulls in.
for h in klu.h amd.h colamd.h btf.h SuiteSparse_config.h; do
    cp -aL "${SRC_INC}/suitesparse/${h}" "${VENDOR}/include/suitesparse/"
done

# --- runtime libraries (single real file per lib, named by SONAME) ---------
# DT_NEEDED entries always reference a library's SONAME (e.g.
# libklu.so.2), never the fully-versioned filename (...so.2.3.1).  So
# we copy the REAL .so but RENAME it to its SONAME — one flat file per lib, no
# symlinks.  This (a) avoids the wheel packer dereferencing a soname symlink
# into a second full-size copy (halves the libgfortran/libgomp footprint) and
# (b) keeps git free of symlink/real-file pairs.
copy_lib() {
    # $1 = soname (e.g. libklu.so.2).
    local soname="$1"
    local real
    real="$(readlink -f "${SRC_LIB}/${soname}")"
    if [[ ! -f "${real}" ]]; then
        echo "ERROR: cannot resolve ${soname} in ${SRC_LIB}" >&2
        exit 1
    fi
    cp "${real}" "${VENDOR}/lib/${soname}"
    chmod u+w "${VENDOR}/lib/${soname}"
}

for soname in \
    libklu.so.2 \
    libamd.so.3 \
    libcolamd.so.3 \
    libbtf.so.2 \
    libsuitesparseconfig.so.7 ; do
    copy_lib "${soname}"
done

# --- GCC runtime deps of the SuiteSparse libs ------------------------------
# libamd  NEEDs libgfortran.so.5 -> {libquadmath.so.0, libgcc_s.so.1}
# libsuitesparseconfig NEEDs libgomp.so.1
# A bare conda env (python only) does NOT ship these, so for a TRULY
# self-contained wheel we bundle them too.  Each carries RPATH=$ORIGIN/. so
# co-located in lib/ the chain resolves with no patching.
for soname in \
    libgfortran.so.5 \
    libquadmath.so.0 \
    libgomp.so.1 \
    libgcc_s.so.1 ; do
    copy_lib "${soname}"
done

echo "Staged libs:"
ls -la "${VENDOR}/lib/"
echo "Vendor tree size: $(du -sh "${VENDOR}" | cut -f1)"
echo "DONE. Commit vendor/ to make the wheel self-contained."
