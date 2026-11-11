#!/usr/bin/env bash
# Static-verify a diff_soup manylinux release wheel WITHOUT installing it.
#
# Confirms the structural invariants of the vendor-prebuilt + auditwheel flow:
#   * exactly ONE coherent SuiteSparse lib tree under klu_ffi/lib/ (no
#     auditwheel double-bundle: no diff_soup.libs/ at all);
#   * the native extensions present (_native_klu and at least one cuDSS
#     _native_solver under cudss_ffi/cu12/ or cudss_ffi/cu13/), and
#     _native_superlu exactly once with DIFF_SOUP_RELEASE_SUPERLU=ON (the
#     default, as in build_release_wheel.sh) or absent with
#     DIFF_SOUP_RELEASE_SUPERLU=OFF;
#   * one OpenMP runtime in the wheel, klu_ffi/lib/libgomp.so.1;
#   * the SuiteSparse libs and libgomp bundled under klu_ffi/lib/ (on x86-64
#     also the vendored tree's libgfortran, libquadmath and libgcc_s), and
#     every library that _native_klu and the bundled libs need either bundled
#     there or a manylinux system library;
#   * _native_klu RPATH == $ORIGIN/lib  (resolves the bundled SuiteSparse libs);
#   * when present, _native_superlu carries SuperLU_MT and its BLAS statically: no SuperLU_MT
#     or BLAS shared library among its DT_NEEDED entries or in the wheel, and
#     no SuperLU_MT or BLAS symbol left undefined; it defines no COLAMD symbol
#     and takes colamd, colamd_recommended and colamd_set_defaults from
#     SuiteSparse's COLAMD; its only non-system dependencies are libgomp.so.1,
#     libcolamd.so.3 and libsuitesparseconfig.so.7, which its $ORIGIN-relative
#     RPATH resolves to the copies in klu_ffi/lib/ that KLU uses, so the two CPU
#     solvers share one OpenMP runtime and one COLAMD;
#   * every _native_solver RPATH is purely $ORIGIN-relative (nvidia pip
#     wheels), no absolute build-toolchain path leaked in;
#   * the wheel carries a manylinux_2_*_x86_64 or manylinux_2_*_aarch64
#     platform tag, not linux_<machine>;
#   * the CUDA libs (libcudss/libcudart/libcublas/libcublasLt/libnvrtc) are NOT bundled.
#
# Usage: [DIFF_SOUP_RELEASE_SUPERLU=ON|OFF] bash verify_release_wheel.sh <wheel.whl>
set -euo pipefail
EXPECT_SUPERLU="${DIFF_SOUP_RELEASE_SUPERLU:-ON}"
case "${EXPECT_SUPERLU}" in
    ON|OFF) ;;
    *) echo "ERROR: DIFF_SOUP_RELEASE_SUPERLU='${EXPECT_SUPERLU}' must be ON or OFF." >&2; exit 1 ;;
esac
WHL="${1:?usage: verify_release_wheel.sh <wheel.whl>}"
WHL="$(cd "$(dirname "${WHL}")" && pwd)/$(basename "${WHL}")"
command -v patchelf >/dev/null || { echo "ERROR: patchelf not on PATH" >&2; exit 1; }
command -v nm >/dev/null || { echo "ERROR: nm (binutils) not on PATH" >&2; exit 1; }

case "$(basename "${WHL}")" in
    *manylinux_2_*_x86_64*.whl) ARCH=x86_64 ;;
    *manylinux_2_*_aarch64*.whl) ARCH=aarch64 ;;
    *) echo "FAIL tag: not manylinux x86_64 or aarch64 ($(basename "${WHL}"))" >&2; exit 1 ;;
esac
echo "OK   tag: manylinux ${ARCH} ($(basename "${WHL}"))"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
( cd "${WORK}" && unzip -q "${WHL}" )

KLU_LIB="${WORK}/diff_soup/_internal/klu_ffi/lib"
KLU="$(ls "${WORK}"/diff_soup/_internal/klu_ffi/_native_klu*.so 2>/dev/null || true)"
SOLVERS="$(ls "${WORK}"/diff_soup/_internal/cudss_ffi/cu*/_native_solver*.so 2>/dev/null || true)"
SUPERLU_ALL="$(find "${WORK}" -name '_native_superlu*.so' | sort)"
SUPERLU="$(ls "${WORK}"/diff_soup/_internal/superlu_ffi/_native_superlu*.so 2>/dev/null || true)"
rc=0

[[ -n "${KLU}" ]]     && echo "OK   _native_klu present"    || { echo "FAIL _native_klu missing" >&2; rc=1; }
[[ -n "${SOLVERS}" ]] && echo "OK   _native_solver present: $(echo "${SOLVERS}" | xargs -n1 dirname | xargs -n1 basename | tr '\n' ' ')" \
    || { echo "FAIL _native_solver missing (no cudss_ffi/cu12 or cu13 build)" >&2; rc=1; }
if [[ "${EXPECT_SUPERLU}" == OFF ]]; then
    if [[ -z "${SUPERLU_ALL}" ]]; then
        echo "OK   _native_superlu absent (DIFF_SOUP_RELEASE_SUPERLU=OFF)"
    else
        echo "FAIL _native_superlu must be absent with DIFF_SOUP_RELEASE_SUPERLU=OFF; found: [${SUPERLU_ALL//"${WORK}"\//}]" >&2; rc=1
    fi
    SUPERLU=""
elif [[ -n "${SUPERLU}" && "$(grep -c . <<<"${SUPERLU_ALL}")" -eq 1 ]]; then
    echo "OK   _native_superlu present once: ${SUPERLU#"${WORK}"/}"
else
    echo "FAIL _native_superlu must be in the wheel exactly once, under superlu_ffi/; found: [${SUPERLU_ALL//"${WORK}"\//}]" >&2; rc=1
    SUPERLU=""
fi

# No auditwheel double-bundle dir.
if find "${WORK}" -maxdepth 1 -name '*.libs' | grep -q .; then
    echo "FAIL auditwheel double-bundle: $(find "${WORK}" -maxdepth 1 -name '*.libs')" >&2; rc=1
else
    echo "OK   no *.libs double-bundle dir"
fi

# Exactly one OpenMP runtime in the wheel: klu_ffi/lib/libgomp.so.1.
GOMP="${KLU_LIB}/libgomp.so.1"
GOMP_ALL="$(find "${WORK}" -name 'libgomp*.so*' | sort)"
if [[ -f "${GOMP}" && "${GOMP_ALL}" == "${GOMP}" ]]; then
    echo "OK   one OpenMP runtime in the wheel: ${GOMP#"${WORK}"/}"
else
    echo "FAIL want exactly one libgomp, klu_ffi/lib/libgomp.so.1; found [${GOMP_ALL//"${WORK}"\//}]" >&2; rc=1
fi

# SuiteSparse and GCC runtime libs bundled exactly once, CUDA libs absent. The
# vendored x86-64 tree is conda-forge's build, whose AMD needs libgfortran; the
# aarch64 source build has no Fortran.
STEMS=(klu amd colamd btf suitesparseconfig gomp)
[[ "${ARCH}" == x86_64 ]] && STEMS+=(gfortran quadmath gcc_s)
for stem in "${STEMS[@]}"; do
    n="$(find "${KLU_LIB}" -name "lib${stem}.so*" 2>/dev/null | wc -l)"
    [[ "${n}" -ge 1 ]] || { echo "FAIL missing bundled lib${stem}" >&2; rc=1; }
done
echo "OK   SuiteSparse and GCC runtime libs bundled under klu_ffi/lib/: $(cd "${KLU_LIB}" 2>/dev/null && ls | tr '\n' ' ')"
# What _native_klu and the bundled libs need is bundled or a manylinux system library.
SYSTEM_RE='^(libc|libm|libpthread|libdl|librt|libgcc_s|libstdc\+\+)\.so\.|^ld-linux-(x86-64|aarch64)\.so\.'
unresolved=""
for so in ${KLU} "${KLU_LIB}"/*.so*; do
    [[ -f "${so}" ]] || continue
    while IFS= read -r lib; do
        [[ -n "${lib}" ]] || continue
        grep -qE "${SYSTEM_RE}" <<<"${lib}" && continue
        [[ -f "${KLU_LIB}/${lib}" ]] || unresolved+=" ${so##*/}->${lib}"
    done < <(patchelf --print-needed "${so}")
done
if [[ -n "${unresolved}" ]]; then
    echo "FAIL needed but neither bundled in klu_ffi/lib/ nor a system library:${unresolved}" >&2; rc=1
else
    echo "OK   every library _native_klu and klu_ffi/lib/ need is bundled there or a system library"
fi
for cuda in libcudss libcudart libcublas libcublasLt libnvrtc; do
    if find "${WORK}" -name "${cuda}.so*" | grep -q .; then
        echo "FAIL CUDA lib bundled (should be nvidia pip dep): ${cuda}" >&2; rc=1
    fi
done
echo "OK   CUDA libs NOT bundled (nvidia pip deps of the cu12 / cu13 extras)"

# RPATHs.
if [[ -n "${KLU}" ]]; then
    r="$(patchelf --print-rpath "${KLU}")"
    [[ "${r}" == '$ORIGIN/lib' ]] && echo "OK   _native_klu RPATH = \$ORIGIN/lib" \
        || { echo "FAIL _native_klu RPATH = '${r}' (want \$ORIGIN/lib)" >&2; rc=1; }
fi
for SOLVER in ${SOLVERS}; do
    r="$(patchelf --print-rpath "${SOLVER}")"
    line="$(basename "$(dirname "${SOLVER}")")"
    if echo "${r}" | tr ':' '\n' | grep -q '^/'; then
        echo "FAIL ${line}/_native_solver RPATH has absolute entry: '${r}'" >&2; rc=1
    else
        echo "OK   ${line}/_native_solver RPATH \$ORIGIN-relative: '${r}'"
    fi
done

# _native_superlu: SuperLU_MT and its BLAS linked in; libgomp and SuiteSparse's
# COLAMD shared with KLU.
if [[ -n "${SUPERLU}" ]]; then
    needed="$(patchelf --print-needed "${SUPERLU}")"
    r="$(patchelf --print-rpath "${SUPERLU}")"
    if grep -qiE 'superlu|blas|lapack' <<<"${needed}" \
        || find "${WORK}" -iname 'libsuperlu*.so*' -o -iname 'libblas*.so*' | grep -q .; then
        echo "FAIL _native_superlu expects a shared SuperLU_MT or BLAS library at run time: NEEDED [$(tr '\n' ' ' <<<"${needed}")]" >&2; rc=1
    else
        echo "OK   no SuperLU_MT or BLAS shared library needed or shipped (NEEDED: $(tr '\n' ' ' <<<"${needed}"))"
    fi
    undef="$(nm -D --undefined-only "${SUPERLU}" | awk '{print $NF}' \
        | grep -E '^(p?dgstr[fs]|pdgstrf_init|get_perm_c|sp_preorder|[sdcz]Create_|Destroy_|Stat(Init|Alloc|Free)|superlu_|SUPERLU_|[sdcz](gemv|gemm|trsv|trsm|ger|axpy|scal|copy|dot|nrm2)_?$|xerbla_)' || true)"
    if [[ -n "${undef}" ]]; then
        echo "FAIL _native_superlu leaves SuperLU_MT/BLAS symbols undefined: $(tr '\n' ' ' <<<"${undef}")" >&2; rc=1
    else
        echo "OK   _native_superlu defines its SuperLU_MT and BLAS symbols (statically linked)"
    fi
    # COLAMD is SuiteSparse's (libcolamd.so.3), not SuperLU_MT's own LGPL copy:
    # the module defines no COLAMD symbol and takes the three it calls from outside.
    colamd_def="$( { nm -D --defined-only "${SUPERLU}"; nm --defined-only "${SUPERLU}" 2>/dev/null || true; } \
        | awk '{print $NF}' | grep -E '^(colamd|symamd)' | sort -u | tr '\n' ' ' || true)"
    colamd_undef="$(nm -D --undefined-only "${SUPERLU}" | awk '{print $NF}' \
        | grep -E '^colamd(_recommended|_set_defaults)?$' | sort | tr '\n' ' ' || true)"
    if [[ -n "${colamd_def}" ]]; then
        echo "FAIL _native_superlu defines COLAMD symbols (SuperLU_MT's own copy linked in): ${colamd_def}" >&2; rc=1
    elif [[ "${colamd_undef}" != "colamd colamd_recommended colamd_set_defaults " ]]; then
        echo "FAIL _native_superlu does not take colamd, colamd_recommended and colamd_set_defaults from a shared COLAMD; undefined: [${colamd_undef}]" >&2; rc=1
    else
        echo "OK   _native_superlu defines no COLAMD symbol and takes ${colamd_undef}from libcolamd.so.3"
    fi
    if tr ':' '\n' <<<"${r}" | grep -q '^/'; then
        echo "FAIL _native_superlu RPATH has absolute entry: '${r}'" >&2; rc=1
    else
        echo "OK   _native_superlu RPATH \$ORIGIN-relative: '${r}'"
    fi
    # Every needed library is a manylinux system library (glibc, libm, libgcc_s,
    # libstdc++) or resolves through the RPATH to a file inside the wheel; the
    # libgomp, libcolamd and libsuitesparseconfig ones are KLU's in klu_ffi/lib/.
    so_dir="$(dirname "${SUPERLU}")"
    while IFS= read -r lib; do
        [[ -n "${lib}" ]] || continue
        case "${lib}" in
            libc.so.*|libm.so.*|libpthread.so.*|libdl.so.*|librt.so.*|ld-linux-x86-64.so.*|ld-linux-aarch64.so.*|libgcc_s.so.*|libstdc++.so.*) continue ;;
        esac
        found=""
        while IFS= read -r entry; do
            [[ -n "${entry}" ]] || continue
            cand="${entry//\$ORIGIN/${so_dir}}/${lib}"
            [[ -f "${cand}" ]] && found="$(cd "$(dirname "${cand}")" && pwd)/${lib}" && break
        done < <(tr ':' '\n' <<<"${r}")
        if [[ -z "${found}" ]]; then
            echo "FAIL _native_superlu needs ${lib}, which neither the system set nor its RPATH inside the wheel provides" >&2; rc=1
        elif [[ "${lib}" == libgomp* || "${lib}" == libcolamd* || "${lib}" == libsuitesparseconfig* ]] \
            && [[ "${found}" != "${KLU_LIB}/${lib}" ]]; then
            echo "FAIL _native_superlu resolves ${lib} to ${found#"${WORK}"/}, not KLU's ${KLU_LIB#"${WORK}"/}/${lib}" >&2; rc=1
        else
            echo "OK   _native_superlu ${lib} -> ${found#"${WORK}"/}"
        fi
    done <<<"${needed}"
    grep -qx 'libgomp.so.1' <<<"${needed}" \
        || { echo "FAIL _native_superlu does not need libgomp.so.1 (want the OpenMP build of SuperLU_MT sharing klu_ffi/lib/libgomp.so.1)" >&2; rc=1; }
    for lib in libcolamd.so.3 libsuitesparseconfig.so.7; do
        grep -qx "${lib}" <<<"${needed}" \
            || { echo "FAIL _native_superlu does not need ${lib} (want SuiteSparse's COLAMD from klu_ffi/lib)" >&2; rc=1; }
    done
fi

[[ "${rc}" -eq 0 ]] && echo "PASS  wheel is a coherent self-contained manylinux artifact" \
    || echo "FAIL  see messages above" >&2
exit "${rc}"
