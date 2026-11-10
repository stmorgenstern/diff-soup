cmake_minimum_required(VERSION 3.18)
project(diff_soup_superlu_ffi LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The SuperLU_MT extension, _native_superlu, is the host sparse LU backend.
# It links SuperLU_MT statically: the OpenMP build of the library
# (libsuperlu_mt_OPENMP.a) and the reference BLAS that SuperLU_MT ships
# (installed as libsuperlu_mt_blas.a). Both are compiled with -fPIC and
# 32-bit indices. SUPERLU_MT_ROOT names the prefix that holds
# include/superlu_mt/ and lib/; it defaults to the active conda prefix.
# build_superlu_mt.sh leaves SuperLU_MT's own COLAMD (SRC/colamd.c, LGPL) out
# of the library (patches/0002), and the extension links SuiteSparse's COLAMD
# instead: libcolamd and libsuitesparseconfig from the SuiteSparse tree the
# KLU extension links (KLU_VENDOR_DIR), shared, found at run time in the KLU
# extension's lib/.
set(SUPERLU_MT_ROOT "$ENV{CONDA_PREFIX}" CACHE PATH
    "Prefix holding include/superlu_mt and lib/libsuperlu_mt_OPENMP.a")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" "-m" "nanobind" "--cmake_dir"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
    RESULT_VARIABLE NANOBIND_CMAKE_DIR_RESULT
)
if(NANOBIND_CMAKE_DIR_RESULT OR NOT NANOBIND_CMAKE_DIR)
    message(FATAL_ERROR "Failed to discover nanobind CMake directory via 'python -m nanobind --cmake_dir'.")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

# The XLA FFI headers ship with jaxlib; jax.ffi.include_dir() is the public
# way to find them.
execute_process(
    COMMAND "${Python_EXECUTABLE}" "-c" "import jax.ffi; print(jax.ffi.include_dir())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE XLA_DIR
    RESULT_VARIABLE XLA_DIR_RESULT
)
if(XLA_DIR_RESULT OR NOT XLA_DIR)
    message(FATAL_ERROR "Failed to discover the XLA FFI include directory from jax.ffi.include_dir().")
endif()

find_path(SUPERLU_MT_INCLUDE_DIR slu_mt_ddefs.h
    HINTS "${SUPERLU_MT_ROOT}/include/superlu_mt"
    NO_DEFAULT_PATH)
find_library(SUPERLU_MT_LIBRARY
    NAMES libsuperlu_mt_OPENMP.a
    HINTS "${SUPERLU_MT_ROOT}/lib"
    NO_DEFAULT_PATH)
find_library(SUPERLU_MT_BLAS_LIBRARY
    NAMES libsuperlu_mt_blas.a
    HINTS "${SUPERLU_MT_ROOT}/lib"
    NO_DEFAULT_PATH)
if(NOT SUPERLU_MT_INCLUDE_DIR OR NOT SUPERLU_MT_LIBRARY OR NOT SUPERLU_MT_BLAS_LIBRARY)
    message(FATAL_ERROR
        "superlu_ffi: SuperLU_MT not found under SUPERLU_MT_ROOT='${SUPERLU_MT_ROOT}'. It needs "
        "include/superlu_mt/slu_mt_ddefs.h, lib/libsuperlu_mt_OPENMP.a and "
        "lib/libsuperlu_mt_blas.a: build SuperLU_MT with THREAD_API=OPENMP, -fPIC and its "
        "internal BLAS, install it into that prefix, or pass -DSUPERLU_MT_ROOT=<prefix>.")
endif()

# COLAMD comes from the SuiteSparse tree the KLU extension links, never from a
# SuperLU_MT library that still carries its own copy.
set(KLU_VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../klu_ffi/vendor" CACHE PATH
    "Tree holding include/suitesparse/klu.h and the SuiteSparse shared libraries in lib/")
get_filename_component(_SLU_SUITESPARSE_LIBDIR "${KLU_VENDOR_DIR}/lib" ABSOLUTE)
set(_SLU_COLAMD "${_SLU_SUITESPARSE_LIBDIR}/libcolamd.so.3")
set(_SLU_SUITESPARSECONFIG "${_SLU_SUITESPARSE_LIBDIR}/libsuitesparseconfig.so.7")
if(NOT EXISTS "${_SLU_COLAMD}" OR NOT EXISTS "${_SLU_SUITESPARSECONFIG}")
    message(FATAL_ERROR
        "superlu_ffi: the SuperLU_MT extension links SuiteSparse's COLAMD, lib/libcolamd.so.3 and "
        "lib/libsuitesparseconfig.so.7 under KLU_VENDOR_DIR='${KLU_VENDOR_DIR}', and at least one is "
        "missing. Pass -DKLU_VENDOR_DIR=<tree> naming the SuiteSparse 7 tree the KLU extension links "
        "(diff_soup/_internal/klu_ffi/build_suitesparse.sh builds one), or build without SuperLU_MT "
        "(-DDIFF_SOUP_BUILD_SUPERLU=OFF).")
endif()
execute_process(
    COMMAND "${CMAKE_NM}" --defined-only "${SUPERLU_MT_LIBRARY}"
    OUTPUT_VARIABLE _SLU_DEFINED
    ERROR_QUIET
    RESULT_VARIABLE _SLU_NM_RESULT
)
if(NOT _SLU_NM_RESULT EQUAL 0)
    message(FATAL_ERROR
        "superlu_ffi: '${CMAKE_NM}' could not list the symbols of ${SUPERLU_MT_LIBRARY}, so the build "
        "cannot tell whether it carries SuperLU_MT's own COLAMD.")
endif()
if(_SLU_DEFINED MATCHES "(^|\n)[0-9a-fA-F]+ [Tt] colamd(\n|$)")
    message(FATAL_ERROR
        "superlu_ffi: ${SUPERLU_MT_LIBRARY} carries SuperLU_MT's own COLAMD (SRC/colamd.c, LGPL), "
        "which the link would take in place of SuiteSparse's. Rebuild SuperLU_MT with "
        "diff_soup/_internal/superlu_ffi/build_superlu_mt.sh, whose patches/0002 leaves it out.")
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)

nanobind_add_module(_native_superlu NOMINSIZE
    native/superlu_solver.cpp
    native/superlu_z_prototypes_check.cpp
)

target_include_directories(_native_superlu PRIVATE
    ${XLA_DIR}
    ${SUPERLU_MT_INCLUDE_DIR}
    # slot_registry.h, the slot registry the three extensions share.
    ${CMAKE_CURRENT_SOURCE_DIR}/../native_common
)
# Select the OpenMP build's declarations in the SuperLU_MT headers.
target_compile_definitions(_native_superlu PRIVATE __OPENMP)

target_link_libraries(_native_superlu PRIVATE
    "${SUPERLU_MT_LIBRARY}"
    "${SUPERLU_MT_BLAS_LIBRARY}"
    "${_SLU_COLAMD}"
    "${_SLU_SUITESPARSECONFIG}"
    OpenMP::OpenMP_CXX
    m
)
target_link_options(_native_superlu PRIVATE "-Wl,-rpath-link,${_SLU_SUITESPARSE_LIBDIR}")

# Keep the static SuperLU_MT and BLAS symbols private to this module, so they
# never interpose on (or get interposed by) another library's BLAS.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(_native_superlu PRIVATE "-Wl,--exclude-libs,ALL" "-Wl,--enable-new-dtags")
endif()

# Link libstdc++ and libgcc statically so the module runs on hosts whose
# system libstdc++ is older than the build compiler's.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(_native_superlu PRIVATE -static-libstdc++ -static-libgcc)
endif()

# The OpenMP runtime and SuiteSparse's COLAMD stay shared dependencies. A
# checkout build finds the OpenMP runtime next to the compiler runtime of the
# build environment and COLAMD in the SuiteSparse tree; an installed module
# finds all of them in the KLU extension's lib/, which bundles them.
set(_slu_build_rpath "")
foreach(_omp_lib ${OpenMP_CXX_LIBRARIES})
    get_filename_component(_omp_dir "${_omp_lib}" DIRECTORY)
    list(APPEND _slu_build_rpath "${_omp_dir}")
endforeach()
list(APPEND _slu_build_rpath "${_SLU_SUITESPARSE_LIBDIR}")
list(REMOVE_DUPLICATES _slu_build_rpath)
set_target_properties(_native_superlu PROPERTIES
    BUILD_RPATH "${_slu_build_rpath}"
    INSTALL_RPATH "\$ORIGIN/../klu_ffi/lib"
)

# In-tree developer build: copy the module next to the Python sources so the
# checkout imports it without a wheel install.
add_custom_command(TARGET _native_superlu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:_native_superlu>"
            "${CMAKE_CURRENT_SOURCE_DIR}/$<TARGET_FILE_NAME:_native_superlu>"
    COMMENT "Copying _native_superlu into diff_soup/_internal/superlu_ffi"
)

install(TARGETS _native_superlu
    LIBRARY DESTINATION diff_soup/_internal/superlu_ffi
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
    DESTINATION diff_soup/_internal/superlu_ffi
)
