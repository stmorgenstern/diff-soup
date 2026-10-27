#!/usr/bin/env bash
# Build the self-contained manylinux diff_soup release wheel for one Python,
# with the KLU extension, the SuperLU_MT extension and the cuDSS extension of
# both CUDA lines (cudss_ffi/cu12 and cudss_ffi/cu13), and tag it manylinux via
# auditwheel.
#
# This is the release path: a prebuilt .whl attached to a GitHub release that
# users `pip install` with no build, no build network and no conda. (The other
# install model is the source build; see README "From a local clone".)
#
# How auditwheel fits in
# ----------------------
# diff_soup vendors the SuiteSparse 7.10.1 stack in-repo
# (diff_soup/_internal/klu_ffi/vendor/). The wheel build already makes the
# SuiteSparse part self-contained: CMake bundles the runtime .so's into
# diff_soup/_internal/klu_ffi/lib/ and sets the _native_klu RPATH to exactly
# $ORIGIN/lib (see this directory's CMakeLists.txt). auditwheel sees those libs
# as already in the wheel and leaves the lib/ tree untouched (no SuiteSparse
# double bundle, no diff_soup.libs/). Its job here is to:
#   (a) report portability (`auditwheel show`), and
#   (b) apply the manylinux_2_*_x86_64 platform tag in place of the
#       non-installable linux_x86_64 tag.
#
# How SuperLU_MT gets in
# ----------------------
# The SuperLU_MT extension (superlu_ffi/_native_superlu) links SuperLU_MT and
# the reference BLAS that ships with it statically (see
# superlu_ffi/CMakeLists.txt), so the wheel carries no SuperLU_MT library and
# the extension needs only libgomp at run time. It shares the OpenMP runtime
# the KLU extension already ships (klu_ffi/lib/libgomp.so.1, which SuiteSparse
# needs): before auditwheel runs, the script sets the _native_superlu RPATH to
# $ORIGIN/../klu_ffi/lib, so auditwheel finds libgomp.so.1 inside the wheel and
# bundles no second copy, and a process that uses both solvers loads a single
# OpenMP runtime. The script builds SuperLU_MT
# from source with diff_soup/_internal/superlu_ffi/build_superlu_mt.sh into a prefix
# under its work directory and passes that prefix to the CUDA 13 line build with
# DIFF_SOUP_BUILD_SUPERLU=ON, so a missing SuperLU_MT fails the build instead of
# silently dropping the extension. The CUDA 12 line build sets
# DIFF_SOUP_BUILD_SUPERLU=OFF, so _native_superlu is built once and reaches the
# release wheel through the CUDA 13 wheel the merge starts from.
#
# The extension goes in only with DIFF_SOUP_RELEASE_SUPERLU=ON. By default
# (OFF) the script builds no SuperLU_MT, both lines build with
# DIFF_SOUP_BUILD_SUPERLU=OFF and the wheel has no _native_superlu: SuperLU_MT
# 1e4bc160 run with more than one OpenMP thread returns a wrong solution, with
# info = 0, in a few factorizations per thousand (its own pdgssv on a
# 1600-unknown nonsymmetric matrix: none in 500 with 1 thread, 5 in 500 with
# 8).
#
# Why the two CUDA lines are built separately and merged
# ------------------------------------------------------
# cudss_ffi/CMakeLists.txt gives the CUDA 12 and CUDA 13 targets the same
# OUTPUT_NAME (_native_solver) in one binary directory, so a single configure
# that builds both lines fails under Ninja ("multiple rules generate
# .../_native_solver.cpython-3XX-x86_64-linux-gnu.so"). The script therefore
# builds each line in its own copy of the source tree (CMake option
# DIFF_SOUP_CUDA_LINES=<line>, kernels compiled by the per-line nvcc command),
# checks that the two single-line wheels differ only in their
# cudss_ffi/cu12 or cudss_ffi/cu13 _native_solver extension (and the CUDA 13
# wheel's _native_superlu), and adds the CUDA 12 extension to the CUDA 13
# wheel.
#
# Pipeline:
#   1. With DIFF_SOUP_RELEASE_SUPERLU=ON, build SuperLU_MT (OpenMP, 32-bit
#      indices, -fPIC, static, with its reference BLAS) with
#      diff_soup/_internal/superlu_ffi/build_superlu_mt.sh into <work dir>/superlu_mt.
#   2. For each CUDA line (13, then 12), copy REPO_ROOT to a temporary
#      directory and run `python -m build --wheel --no-isolation` there with
#        -C cmake.define.DIFF_SOUP_CUDA_LINES=<line>
#        -C cmake.define.DIFF_SOUP_CUDA_KERNEL_BUILD=command
#        -C cmake.define.DIFF_SOUP_BUILD_KLU=ON
#      plus, for the CUDA 13 line with DIFF_SOUP_RELEASE_SUPERLU=ON,
#        -C cmake.define.DIFF_SOUP_BUILD_SUPERLU=ON
#        -C cmake.define.SUPERLU_MT_ROOT=<work dir>/superlu_mt
#      and otherwise -C cmake.define.DIFF_SOUP_BUILD_SUPERLU=OFF.
#      REPO_ROOT itself is never built in. Then merge the cu12 _native_solver
#      into the cu13 wheel, set the _native_superlu RPATH (when built) to
#      $ORIGIN/../klu_ffi/lib (the shared libgomp.so.1) and repack it as
#      diff_soup-<ver>-cp3XX-cp3XX-linux_x86_64.whl.
#      The build runs without isolation because the nvidia wheels of the two
#      lines are not in [build-system].requires: each line pins a different
#      cuda-toolkit major, so pip cannot resolve both at once and an isolated
#      build environment holds neither. The build environment below holds them,
#      and its pinned jaxlib 0.11 is the build ABI (the XLA FFI C ABI of the
#      build must match the runtime).
#   3. `auditwheel show`: portability report (expect external references only to
#      system glibc/libm/libgcc_s/libquadmath/libgomp, libcudss.so.1 and
#      libcudart.so.12 / libcudart.so.13; the SuiteSparse libs and libgomp are
#      in the wheel).
#   4. `auditwheel repair` with the CUDA libs of both lines excluded (libcudss,
#      libcudart, libcublas, libcublasLt and libnvrtc stay nvidia pip-wheel
#      dependencies, not bundled). This applies the manylinux tag and bundles
#      nothing: the SuiteSparse libs and libgomp are already in the wheel.
#   5. RPATH polish: drop any absolute RPATH entry (such as a build toolchain's
#      library directory) from each _native_solver, so every extension carries
#      only $ORIGIN-relative RPATHs. The cuDSS extension links libstdc++/libgcc
#      statically, so such an entry is never load-bearing.
#   The script ends by running verify_release_wheel.sh (next to this script)
#   and `auditwheel show` on the final wheel.
#
# Build environment
# -----------------
# Use a dedicated build environment, not the development one. Nothing is
# installed for you. It holds:
#   * build, auditwheel, patchelf, wheel, cmake, ninja;
#   * the [build-system].requires: scikit-build-core, nanobind and the pinned
#     jax/jaxlib 0.11 line;
#   * the CUDA 13 line: nvidia-cudss-cu13, nvidia-cuda-runtime, nvidia-cublas
#     and nvidia-cuda-nvcc (its nvcc compiles the CUDA 13 kernels);
#   * the CUDA 12 line: nvidia-cudss-cu12, nvidia-cuda-runtime-cu12 and
#     nvidia-cublas-cu12, plus nvidia-cuda-nvcc-cu12 for the CUDA 12 headers
#     (it ships no nvcc binary). These go in with --no-deps, since their
#     cuda-toolkit pins conflict with the CUDA 13 line's;
#   * a CUDA 12 toolkit nvcc (for example the CUDA 12.9 redistributable
#     cuda_nvcc archive), named by the CUDACXX environment variable, which
#     cudss_ffi/CMakeLists.txt reads when the nvcc wheel of a line has none.
# The SuperLU_MT build also needs a C compiler with OpenMP, make, and git to
# fetch the SuperLU_MT sources (not needed when SUPERLU_MT_SOURCE names a
# checkout); the manylinux image provides all three.
#
#   python3.12 -m venv /work/venv-cp312 && source /work/venv-cp312/bin/activate
#   pip install build auditwheel patchelf wheel cmake ninja \
#       'scikit-build-core>=0.10' 'nanobind>=2.0' 'jax>=0.11.1,<0.12' 'jaxlib>=0.11.1,<0.12'
#   pip install 'nvidia-cudss-cu13>=1.0.0,<2' 'nvidia-cuda-runtime>=13,<14' \
#       'nvidia-cublas>=13,<14' 'nvidia-cuda-nvcc>=13,<14'
#   pip install --no-deps 'nvidia-cudss-cu12>=1.0.0,<2' nvidia-cuda-runtime-cu12 \
#       nvidia-cublas-cu12 'nvidia-cuda-nvcc-cu12==<release of the CUDA 12 nvcc>'
#   export CUDACXX=/work/nvcc12/bin/nvcc
#
# Release builds run inside the quay.io/pypa/manylinux_2_28_x86_64 container,
# whose glibc and GCC set the manylinux tag. Put the repository, the build
# environment and the CUDA 12 nvcc under one directory, bind-mount it, and run
# as your own user so the outputs stay yours:
#
#   docker run --rm -u "$(id -u):$(id -g)" -e HOME=/work/home \
#       -v <host dir>:/work quay.io/pypa/manylinux_2_28_x86_64 \
#       bash -c 'source /work/venv-cp312/bin/activate &&
#                export CUDACXX=/work/nvcc12/bin/nvcc &&
#                bash /work/src/diff_soup/_internal/klu_ffi/build_release_wheel.sh \
#                    /work/src /work/out/cp312'
#
# Create the environment with /opt/python/cp312-cp312/bin/python inside the
# container (and once more per Python version to release). Build from a clean
# tree, such as a `git archive` export: the whole of REPO_ROOT is copied for
# each line.
#
# Usage:
#   bash build_release_wheel.sh [REPO_ROOT] [OUTDIR]
# Defaults: REPO_ROOT = three levels up from this script;
#           OUTDIR = REPO_ROOT/dist_release.
# Env knobs:
#   AUDITWHEEL_PLAT  manylinux platform tag to request (default manylinux_2_28_x86_64)
#   CUDACXX          path to the CUDA 12 toolkit nvcc used for the CUDA 12 line
#                    (needed unless a CUDA 12 nvcc is first on PATH)
#   DIFF_SOUP_RELEASE_SUPERLU  ON to build SuperLU_MT and ship _native_superlu,
#                    OFF (default) to leave it out; verify_release_wheel.sh
#                    reads the same variable
#   SUPERLU_MT_SOURCE  an existing SuperLU_MT checkout to build instead of
#                    cloning it (passed to
#                    diff_soup/_internal/superlu_ffi/build_superlu_mt.sh, as are
#                    SUPERLU_MT_COMMIT and CC)
#   TMPDIR           where the source copies, the SuperLU_MT build and the
#                    intermediate wheels go (needs a few GB; removed on exit)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${1:-${SCRIPT_DIR}/../../..}" && pwd)"
OUTDIR="${2:-${REPO_ROOT}/dist_release}"
PLAT="${AUDITWHEEL_PLAT:-manylinux_2_28_x86_64}"
SUPERLU_MT_BUILD="${REPO_ROOT}/diff_soup/_internal/superlu_ffi/build_superlu_mt.sh"
SUPERLU="${DIFF_SOUP_RELEASE_SUPERLU:-OFF}"
case "${SUPERLU}" in
    ON|OFF) ;;
    *) echo "ERROR: DIFF_SOUP_RELEASE_SUPERLU='${SUPERLU}' must be ON or OFF." >&2; exit 1 ;;
esac

TOOLS=(python auditwheel patchelf)
[[ "${SUPERLU}" == ON ]] && TOOLS+=(make "${CC:-gcc}")
for tool in "${TOOLS[@]}"; do
    command -v "${tool}" >/dev/null 2>&1 || {
        echo "ERROR: '${tool}' is not on PATH; activate the release build environment described in this script's header." >&2
        exit 1
    }
done
if [[ "${SUPERLU}" == ON ]]; then
    if [[ -z "${SUPERLU_MT_SOURCE:-}" ]] && ! command -v git >/dev/null 2>&1; then
        echo "ERROR: 'git' is not on PATH and SUPERLU_MT_SOURCE is unset; install git or set SUPERLU_MT_SOURCE to a SuperLU_MT checkout." >&2
        exit 1
    fi
    if [[ ! -f "${SUPERLU_MT_BUILD}" ]]; then
        echo "ERROR: ${SUPERLU_MT_BUILD} is missing; build from a complete diff_soup source tree." >&2
        exit 1
    fi
fi
for module in build wheel; do
    python -m "${module}" --help >/dev/null 2>&1 || {
        echo "ERROR: 'python -m ${module}' is unavailable; install it in the release build environment (pip install ${module})." >&2
        exit 1
    }
done
if [[ -n "${CUDACXX:-}" && ! -x "${CUDACXX}" ]]; then
    echo "ERROR: CUDACXX='${CUDACXX}' is not an executable file; set it to the CUDA 12 toolkit nvcc." >&2
    exit 1
fi

mkdir -p "${OUTDIR}"
OUTDIR="$(cd "${OUTDIR}" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# The single entry of a glob that must match exactly one path.
one() {
    if [[ $# -ne 1 || ! -e "$1" ]]; then
        echo "ERROR: expected exactly one match, found: $*" >&2
        exit 1
    fi
    printf '%s\n' "$1"
}

SUPERLU_PREFIX="${WORK}/superlu_mt"
if [[ "${SUPERLU}" == ON ]]; then
    echo "==> [1/5] Building SuperLU_MT (static, OpenMP, its reference BLAS) with diff_soup/_internal/superlu_ffi/build_superlu_mt.sh"
    BUILD_DIR="${WORK}/superlu_mt-build" bash "${SUPERLU_MT_BUILD}" "${SUPERLU_PREFIX}"
    for f in lib/libsuperlu_mt_OPENMP.a lib/libsuperlu_mt_blas.a include/superlu_mt/slu_mt_ddefs.h; do
        if [[ ! -f "${SUPERLU_PREFIX}/${f}" ]]; then
            echo "ERROR: the SuperLU_MT build left no ${f} under ${SUPERLU_PREFIX}." >&2
            exit 1
        fi
    done
    rm -rf "${WORK}/superlu_mt-build"
else
    echo "==> [1/5] SuperLU_MT left out (DIFF_SOUP_RELEASE_SUPERLU=OFF)"
fi

echo "==> [2/5] Building one wheel per CUDA line (build isolation OFF; vendored SuiteSparse, no conda)"
for LINE in 13 12; do
    SRC="${WORK}/src-cu${LINE}"
    cp -a "${REPO_ROOT}" "${SRC}"
    # _native_superlu is built once, in the CUDA 13 line whose wheel the merge
    # below starts from.
    if [[ "${LINE}" == 13 && "${SUPERLU}" == ON ]]; then
        SUPERLU_DEFINES=(-C cmake.define.DIFF_SOUP_BUILD_SUPERLU=ON
                         -C cmake.define.SUPERLU_MT_ROOT="${SUPERLU_PREFIX}")
    else
        SUPERLU_DEFINES=(-C cmake.define.DIFF_SOUP_BUILD_SUPERLU=OFF)
    fi
    echo "    CUDA ${LINE}: building in ${SRC} (${SUPERLU_DEFINES[*]})"
    python -m build --wheel --no-isolation --outdir "${WORK}/raw/cu${LINE}" \
        -C cmake.define.DIFF_SOUP_CUDA_LINES="${LINE}" \
        -C cmake.define.DIFF_SOUP_CUDA_KERNEL_BUILD=command \
        -C cmake.define.DIFF_SOUP_BUILD_KLU=ON \
        "${SUPERLU_DEFINES[@]}" \
        "${SRC}"
    rm -rf "${SRC}"
done

echo "    merging the CUDA 12 extension into the CUDA 13 wheel"
for LINE in 13 12; do
    mkdir -p "${WORK}/unpack/cu${LINE}"
    WHL="$(one "${WORK}/raw/cu${LINE}"/diff_soup-*-linux_x86_64.whl)"
    ( cd "${WORK}/unpack/cu${LINE}" && python -m wheel unpack "${WHL}" >/dev/null )
done
TREE13="$(one "${WORK}"/unpack/cu13/diff_soup-*)"
TREE12="$(one "${WORK}"/unpack/cu12/diff_soup-*)"
( cd "${TREE13}" && find . -type f ! -name RECORD | sort ) > "${WORK}/files-cu13"
( cd "${TREE12}" && find . -type f ! -name RECORD | sort ) > "${WORK}/files-cu12"
ONLY13="$(comm -23 "${WORK}/files-cu13" "${WORK}/files-cu12")"
ONLY12="$(comm -13 "${WORK}/files-cu13" "${WORK}/files-cu12")"
SO13_RE='^\./diff_soup/_internal/cudss_ffi/cu13/_native_solver[^/]*\.so$'
SO12_RE='^\./diff_soup/_internal/cudss_ffi/cu12/_native_solver[^/]*\.so$'
SLU_RE='^\./diff_soup/_internal/superlu_ffi/_native_superlu[^/]*\.so$'
N_SLU=0
[[ "${SUPERLU}" == ON ]] && N_SLU=1
if [[ "$(grep -c . <<<"${ONLY13}")" -ne $((1 + N_SLU)) || "$(grep -c . <<<"${ONLY12}")" -ne 1 ]] \
    || [[ "$(grep -cE "${SO13_RE}" <<<"${ONLY13}")" -ne 1 ]] \
    || [[ "$(grep -cE "${SLU_RE}" <<<"${ONLY13}")" -ne "${N_SLU}" ]] \
    || ! grep -qE "${SO12_RE}" <<<"${ONLY12}"; then
    if [[ "${SUPERLU}" == ON ]]; then
        WANT13="cudss_ffi/cu13/_native_solver*.so and superlu_ffi/_native_superlu*.so"
    else
        WANT13="cudss_ffi/cu13/_native_solver*.so"
    fi
    echo "ERROR: the single-line wheels must differ only in ${WANT13} (CUDA 13 wheel) and cudss_ffi/cu12/_native_solver*.so (CUDA 12 wheel); only in the CUDA 13 wheel: [${ONLY13}]; only in the CUDA 12 wheel: [${ONLY12}]." >&2
    exit 1
fi
if grep -q '_native_superlu' "${WORK}/files-cu12" || { [[ "${SUPERLU}" == OFF ]] && grep -q '_native_superlu' "${WORK}/files-cu13"; }; then
    echo "ERROR: _native_superlu was built with DIFF_SOUP_BUILD_SUPERLU=OFF." >&2
    exit 1
fi
cp "${TREE12}/${ONLY12#./}" "${TREE13}/${ONLY12#./}"
echo "    added ${ONLY12#./}"
if [[ "${SUPERLU}" == ON ]]; then
    SLU_SO="${TREE13}/$(grep -E "${SLU_RE}" <<<"${ONLY13}" | cut -c3-)"
    echo "    kept ${SLU_SO#"${TREE13}"/} from the CUDA 13 wheel"
    # One OpenMP runtime per process: _native_superlu resolves libgomp.so.1 to
    # the copy the KLU extension ships in klu_ffi/lib/, so auditwheel sees it as
    # in-wheel and bundles no second libgomp.
    if [[ ! -f "${TREE13}/diff_soup/_internal/klu_ffi/lib/libgomp.so.1" ]]; then
        echo "ERROR: the CUDA 13 wheel has no diff_soup/_internal/klu_ffi/lib/libgomp.so.1 for _native_superlu to share; build with DIFF_SOUP_BUILD_KLU=ON." >&2
        exit 1
    fi
    if ! patchelf --print-needed "${SLU_SO}" | grep -qx 'libgomp.so.1'; then
        echo "ERROR: ${SLU_SO#"${TREE13}"/} must need libgomp.so.1 (GCC's OpenMP runtime); NEEDED: $(patchelf --print-needed "${SLU_SO}" | tr '\n' ' ')." >&2
        exit 1
    fi
    patchelf --force-rpath --set-rpath '$ORIGIN/../klu_ffi/lib' "${SLU_SO}"
    echo "    superlu_ffi/_native_superlu RPATH -> \$ORIGIN/../klu_ffi/lib (shares klu_ffi/lib/libgomp.so.1)"
fi
mkdir -p "${WORK}/raw/merged"
( cd "${WORK}/unpack/cu13" && python -m wheel pack "$(basename "${TREE13}")" --dest-dir "${WORK}/raw/merged" >/dev/null )
RAW_WHL="$(one "${WORK}"/raw/merged/diff_soup-*-linux_x86_64.whl)"
echo "    raw wheel: ${RAW_WHL}"

echo "==> [3/5] auditwheel show (portability report; SuiteSparse expected in-wheel)"
auditwheel show "${RAW_WHL}"

echo "==> [4/5] auditwheel repair (plat=${PLAT}; exclude CUDA libs; no SuiteSparse or libgomp re-bundle)"
auditwheel repair \
    --plat "${PLAT}" \
    --exclude libcudss.so.1 \
    --exclude libcudart.so.12 \
    --exclude libcublas.so.12 \
    --exclude libcublasLt.so.12 \
    --exclude libnvrtc.so.12 \
    --exclude libcudart.so.13 \
    --exclude libcublas.so.13 \
    --exclude libcublasLt.so.13 \
    --exclude libnvrtc.so.13 \
    -w "${WORK}/repaired" \
    "${RAW_WHL}"
REP_WHL="$(one "${WORK}"/repaired/diff_soup-*manylinux*.whl)"
echo "    repaired wheel: ${REP_WHL}"

echo "==> [5/5] RPATH polish: drop absolute RPATH entries from the cuDSS extensions"
mkdir -p "${WORK}/polish"
( cd "${WORK}/polish" && python -m wheel unpack "${REP_WHL}" >/dev/null )
PKG_DIR="$(one "${WORK}"/polish/diff_soup-*)"
REPACK=0
for SOLVER in "${PKG_DIR}"/diff_soup/_internal/cudss_ffi/cu*/_native_solver*.so; do
    # Keep only the $ORIGIN-relative nvidia entries; drop any absolute (/...) entry.
    OLD_RPATH="$(patchelf --print-rpath "${SOLVER}")"
    NEW_RPATH="$(python - "${OLD_RPATH}" <<'PY'
import sys
entries = [e for e in sys.argv[1].split(":") if e and not e.startswith("/")]
print(":".join(entries))
PY
)"
    if [[ "${OLD_RPATH}" != "${NEW_RPATH}" ]]; then
        patchelf --force-rpath --set-rpath "${NEW_RPATH}" "${SOLVER}"
        echo "    $(basename "$(dirname "${SOLVER}")")/_native_solver RPATH: ${OLD_RPATH}"
        echo "                       -> ${NEW_RPATH}"
        REPACK=1
    else
        echo "    $(basename "$(dirname "${SOLVER}")")/_native_solver RPATH already clean (${OLD_RPATH})."
    fi
done
if [[ "${REPACK}" -eq 1 ]]; then
    ( cd "${WORK}/polish" && python -m wheel pack "$(basename "${PKG_DIR}")" --dest-dir "${WORK}/repaired" >/dev/null )
    echo "    repacked the relocatable wheel"
fi
FINAL_WHL="${OUTDIR}/$(basename "${REP_WHL}")"
cp "${REP_WHL}" "${FINAL_WHL}"

echo "==> Verify: ${FINAL_WHL}"
bash "${SCRIPT_DIR}/verify_release_wheel.sh" "${FINAL_WHL}"
auditwheel show "${FINAL_WHL}"

VERSION="$(basename "${FINAL_WHL}" | cut -d- -f2)"
echo
echo "==> Done. Self-contained manylinux release wheel:"
echo "    ${FINAL_WHL}"
echo "    Attach to a GitHub release:"
echo "      gh release create v${VERSION} ${FINAL_WHL} --title 'diff_soup ${VERSION}'"
