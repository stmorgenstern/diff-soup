cmake_minimum_required(VERSION 3.18)
project(diff_soup_cudss_ffi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The cuDSS extension is built once per CUDA line from the C++ source
# native/single_solve.cpp and the CUDA sources native/status_kernels.cu and
# native/assembly_kernels.cu. The
# XLA FFI C API it registers against is CUDA version independent (the stream
# is an opaque handle), so the two builds differ only in the NVIDIA wheels
# they take headers, the CUDA compiler and libraries from:
#
#   CUDA 12: nvidia-cudss-cu12, nvidia-cuda-runtime-cu12 (one package
#            directory each under nvidia/), and a CUDA 12 nvcc: the
#            nvidia-cuda-nvcc-cu12 wheel ships ptxas but no bin/nvcc, so the
#            compiler comes from DIFF_SOUP_NVCC, $CUDACXX or PATH (below)
#   CUDA 13: nvidia-cudss-cu13, nvidia-cuda-runtime, nvidia-cuda-nvcc
#            (all under nvidia/cu13/)
#
# Both lines need cuDSS 1.0 or newer (nvidia-cudss-cu12 or nvidia-cudss-cu13
# 1.x): the configure step reads the version from the resolved cudss.h and
# stops below 1.0, and the extension links libcudss.so.1.
#
# Each build lands in its own sub-package, diff_soup/_internal/cudss_ffi/cu12
# or .../cu13, and the loader in diff_soup/linear/_cudss/_ffi_loader.py picks
# the one matching the installed jax CUDA plugin at import time.
#
# DIFF_SOUP_CUDA_LINES selects the lines to build: "auto" (default) builds
# every line whose wheels are installed in the build environment and fails if
# there is none; "12", "13", or "12;13" require those lines.
set(DIFF_SOUP_CUDA_LINES "auto" CACHE STRING
    "CUDA lines to build the cuDSS extension for: auto, 12, 13, or 12;13")

# DIFF_SOUP_CUDA_ARCHITECTURES lists the GPU architectures the extension's CUDA
# kernels are compiled for: machine code for compute capability 8.0 (A100), 8.9
# (L40S, RTX 40) and 9.0 (H100), plus PTX of 9.0 that the driver compiles for
# newer GPUs. DIFF_SOUP_CUDA_KERNEL_BUILD selects how they are compiled: "auto"
# uses CMake's CUDA language when one line is built and one custom nvcc command
# per line when two are; "command" always uses the custom commands.
set(DIFF_SOUP_CUDA_ARCHITECTURES "80-real;89-real;90-real;90-virtual" CACHE STRING
    "GPU architectures of the cuDSS extension's CUDA kernels (CUDA_ARCHITECTURES syntax)")
set(DIFF_SOUP_CUDA_KERNEL_BUILD "auto" CACHE STRING
    "How the cuDSS extension's CUDA kernels are compiled: auto or command")

# DIFF_SOUP_NVCC names the nvcc of a CUDA line whose nvcc wheel ships no
# bin/nvcc (the CUDA 12 nvidia-cuda-nvcc-cu12 wheel holds ptxas only). A line
# takes its wheel's bin/nvcc when there is one; otherwise the first of
# DIFF_SOUP_NVCC, $CUDACXX and nvcc on PATH whose CUDA major is the line's.
set(DIFF_SOUP_NVCC "" CACHE FILEPATH
    "nvcc for a CUDA line whose nvcc wheel ships no bin/nvcc (a CUDA toolkit's bin/nvcc)")

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

# Optional override of the cuDSS include dir + .so to build against a
# locally staged cuDSS without touching the installed nvidia-cudss package.
# Applies to every line built. Pass
#   -DCUDSS_INCLUDE_OVERRIDE=<dir> -DCUDSS_LIBRARY_OVERRIDE=<libcudss.so.N>
set(CUDSS_INCLUDE_OVERRIDE "" CACHE PATH "Override the cuDSS include directory")
set(CUDSS_LIBRARY_OVERRIDE "" CACHE FILEPATH "Override the cuDSS shared library")

# Resolve the NVIDIA wheel headers and libraries for one CUDA line. Prints
# KEY=VALUE lines, or exits non-zero when that line's wheels are absent.
set(_NVIDIA_RESOLVER "
import sys
from importlib.util import find_spec
from pathlib import Path

line = sys.argv[1]

def base(mod):
    spec = find_spec(mod)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(f'package not installed: {mod}')
    return Path(spec.submodule_search_locations[0])

def one(path, pattern):
    hits = sorted(path.glob(pattern), key=lambda p: len(str(p)))
    if not hits:
        raise RuntimeError(f'no match for {pattern} in {path}')
    return hits[-1]

try:
    if line == '12':
        cudss, cudart = base('nvidia.cu12'), base('nvidia.cuda_runtime')
        try:
            nvcc = base('nvidia.cuda_nvcc')
        except RuntimeError:
            nvcc = None
        rpath = ['nvidia/cu12/lib', 'nvidia/cuda_runtime/lib']
    elif line == '13':
        cudss = cudart = nvcc = base('nvidia.cu13')
        rpath = ['nvidia/cu13/lib']
    else:
        raise RuntimeError(f'unknown CUDA line {line!r}')
    print(f'CUDSS_INCLUDE={cudss / \"include\"}')
    print(f'CUDSS_LIBRARY={one(cudss / \"lib\", \"libcudss.so*\")}')
    print(f'CUDART_INCLUDE={cudart / \"include\"}')
    print(f'CUDART_LIBRARY={one(cudart / \"lib\", \"libcudart.so*\")}')
    nvcc_include = nvcc / 'include' if nvcc is not None else None
    print('NVCC_INCLUDE=' + (str(nvcc_include) if nvcc_include and nvcc_include.is_dir() else ''))
    nvcc_bin = nvcc / 'bin' / 'nvcc' if nvcc is not None else None
    print('NVCC_EXECUTABLE=' + (str(nvcc_bin) if nvcc_bin and nvcc_bin.is_file() else ''))
    print('WHEEL_RPATH=' + ';'.join(rpath))
except Exception as exc:
    sys.exit(f'CUDA {line} wheels not usable: {exc}')
")

function(_resolve_cuda_line LINE OUT_OK)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "${_NVIDIA_RESOLVER}" "${LINE}"
        OUTPUT_VARIABLE _out
        ERROR_VARIABLE _err
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _rc
    )
    if(_rc)
        set(${OUT_OK} FALSE PARENT_SCOPE)
        set(_RESOLVE_ERROR "${_err}" PARENT_SCOPE)
        return()
    endif()
    foreach(key CUDSS_INCLUDE CUDSS_LIBRARY CUDART_INCLUDE CUDART_LIBRARY
                NVCC_INCLUDE NVCC_EXECUTABLE WHEEL_RPATH)
        string(REGEX MATCH "${key}=([^\n]*)" _ "${_out}")
        set(CU${LINE}_${key} "${CMAKE_MATCH_1}" PARENT_SCOPE)
    endforeach()
    set(${OUT_OK} TRUE PARENT_SCOPE)
endfunction()

# The nvcc for line LINE when its nvcc wheel ships none: the first of DIFF_SOUP_NVCC, $CUDACXX
# and nvcc on PATH whose `nvcc --version` reports CUDA major LINE. OUT is empty when none
# does; OUT_NOTE then says what each candidate was.
function(_diff_soup_line_nvcc LINE OUT OUT_NOTE)
    find_program(DIFF_SOUP_NVCC_ON_PATH nvcc)
    mark_as_advanced(DIFF_SOUP_NVCC_ON_PATH)
    set(_notes "")
    foreach(_source DIFF_SOUP_NVCC CUDACXX PATH)
        if(_source STREQUAL "DIFF_SOUP_NVCC")
            set(_candidate "${DIFF_SOUP_NVCC}")
        elseif(_source STREQUAL "CUDACXX")
            set(_candidate "$ENV{CUDACXX}")
        else()
            set(_candidate "${DIFF_SOUP_NVCC_ON_PATH}")
        endif()
        if(NOT _candidate OR _candidate MATCHES "-NOTFOUND$")
            list(APPEND _notes "${_source}: none")
            continue()
        endif()
        if(NOT EXISTS "${_candidate}" OR IS_DIRECTORY "${_candidate}")
            list(APPEND _notes "${_source}: ${_candidate} is not a file")
            continue()
        endif()
        execute_process(COMMAND "${_candidate}" --version
            OUTPUT_VARIABLE _version_text ERROR_QUIET RESULT_VARIABLE _version_rc)
        if(_version_rc OR NOT _version_text MATCHES "release ([0-9]+)\\.([0-9]+)")
            list(APPEND _notes "${_source}: ${_candidate} reports no CUDA release")
            continue()
        endif()
        if(NOT CMAKE_MATCH_1 EQUAL LINE)
            list(APPEND _notes
                 "${_source}: ${_candidate} is CUDA ${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
            continue()
        endif()
        set(${OUT} "${_candidate}" PARENT_SCOPE)
        set(${OUT_NOTE} "${_source}: ${_candidate} (CUDA ${CMAKE_MATCH_1}.${CMAKE_MATCH_2})"
            PARENT_SCOPE)
        return()
    endforeach()
    list(JOIN _notes "; " _note)
    set(${OUT} "" PARENT_SCOPE)
    set(${OUT_NOTE} "${_note}" PARENT_SCOPE)
endfunction()

if(DIFF_SOUP_CUDA_LINES STREQUAL "auto")
    set(_requested 12 13)
    set(_required FALSE)
else()
    set(_requested ${DIFF_SOUP_CUDA_LINES})
    set(_required TRUE)
endif()

set(_resolved_lines "")
foreach(LINE ${_requested})
    if(NOT LINE MATCHES "^(12|13)$")
        message(FATAL_ERROR "DIFF_SOUP_CUDA_LINES accepts 12, 13, 12;13, or auto; got '${LINE}'.")
    endif()
    _resolve_cuda_line(${LINE} _ok)
    if(NOT _ok)
        if(_required)
            message(FATAL_ERROR "cudss_ffi: cannot build the CUDA ${LINE} extension: ${_RESOLVE_ERROR}")
        endif()
        message(STATUS "cudss_ffi: CUDA ${LINE} wheels not installed; skipping that line.")
        continue()
    endif()
    if(NOT CU${LINE}_NVCC_EXECUTABLE)
        _diff_soup_line_nvcc(${LINE} _line_nvcc _line_nvcc_note)
        if(NOT _line_nvcc)
            string(CONCAT _nvcc_options
                "pass -DDIFF_SOUP_NVCC=<path to a CUDA ${LINE} nvcc> (pip: -C "
                "cmake.define.DIFF_SOUP_NVCC=<path>), set CUDACXX to that nvcc, or put it "
                "on PATH")
            if(_required)
                message(FATAL_ERROR
                    "cudss_ffi: cannot build the CUDA ${LINE} extension: it needs a CUDA ${LINE} "
                    "nvcc, the line's wheels provide no bin/nvcc (nvidia-cuda-nvcc-cu12 never "
                    "does; nvidia-cuda-nvcc does for CUDA 13), and no other source holds one "
                    "(${_line_nvcc_note}); ${_nvcc_options}.")
            endif()
            message(STATUS
                "cudss_ffi: no CUDA ${LINE} nvcc (${_line_nvcc_note}); skipping that line. "
                "To build it, ${_nvcc_options}.")
            continue()
        endif()
        message(STATUS "cudss_ffi: CUDA ${LINE} kernels use the nvcc of ${_line_nvcc_note}")
        set(CU${LINE}_NVCC_EXECUTABLE "${_line_nvcc}")
    endif()
    list(APPEND _resolved_lines ${LINE})
endforeach()

# Host flags nvcc needs to accept the C++ compiler: none when nvcc supports its version,
# -allow-unsupported-compiler when nvcc refuses the version but compiles with it.
function(_diff_soup_nvcc_host_flags NVCC OUT)
    set(_probe "${CMAKE_CURRENT_BINARY_DIR}/nvcc_host_probe.cu")
    file(WRITE "${_probe}" "__global__ void probe() {}\nint main() { return 0; }\n")
    execute_process(
        COMMAND "${NVCC}" -ccbin "${CMAKE_CXX_COMPILER}" -c "${_probe}" -o "${_probe}.o"
        RESULT_VARIABLE _rc OUTPUT_QUIET ERROR_VARIABLE _err)
    if(NOT _rc)
        set(${OUT} "" PARENT_SCOPE)
        return()
    endif()
    execute_process(
        COMMAND "${NVCC}" -ccbin "${CMAKE_CXX_COMPILER}" -allow-unsupported-compiler
                -c "${_probe}" -o "${_probe}.o"
        RESULT_VARIABLE _rc OUTPUT_QUIET ERROR_VARIABLE _err_unsupported)
    if(_rc)
        message(FATAL_ERROR
            "cudss_ffi: ${NVCC} cannot compile CUDA with the host compiler "
            "${CMAKE_CXX_COMPILER}:\n${_err}")
    endif()
    message(STATUS
        "cudss_ffi: ${NVCC} does not list ${CMAKE_CXX_COMPILER_ID} "
        "${CMAKE_CXX_COMPILER_VERSION} as supported; compiling the status kernels with "
        "-allow-unsupported-compiler")
    set(${OUT} "-allow-unsupported-compiler" PARENT_SCOPE)
endfunction()

# Kernel build: with one line, CMake's CUDA language on that line's nvcc; with two lines in one
# configure (CMake holds one CUDA toolchain per build tree), each line's object is compiled by
# a custom command with its own nvcc. DIFF_SOUP_CUDA_KERNEL_BUILD=command forces the second
# mechanism for one line too.
list(LENGTH _resolved_lines _line_count)
if(_line_count EQUAL 1 AND NOT DIFF_SOUP_CUDA_KERNEL_BUILD STREQUAL "command")
    set(_kernel_build "language")
    list(GET _resolved_lines 0 _line)
    if(NOT CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER "${CU${_line}_NVCC_EXECUTABLE}")
    endif()
    if(NOT CMAKE_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
    endif()
    _diff_soup_nvcc_host_flags("${CMAKE_CUDA_COMPILER}" _host_flags)
    if(_host_flags)
        string(APPEND CMAKE_CUDA_FLAGS " ${_host_flags}")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${DIFF_SOUP_CUDA_ARCHITECTURES}")
    enable_language(CUDA)
    message(STATUS "cudss_ffi: CUDA kernels compiled by ${CMAKE_CUDA_COMPILER} "
                   "(CUDA ${CMAKE_CUDA_COMPILER_VERSION}) for ${DIFF_SOUP_CUDA_ARCHITECTURES}")
else()
    set(_kernel_build "command")
endif()

# The objects of the CUDA sources (native/status_kernels.cu, native/assembly_kernels.cu) for
# CUDA line LINE, compiled by that line's nvcc for DIFF_SOUP_CUDA_ARCHITECTURES with the C++
# compiler as host compiler.
function(_diff_soup_cuda_kernel_objects LINE OUT)
    set(_nvcc "${CU${LINE}_NVCC_EXECUTABLE}")
    _diff_soup_nvcc_host_flags("${_nvcc}" _host_flags)
    set(_gencode "")
    foreach(_arch ${DIFF_SOUP_CUDA_ARCHITECTURES})
        if(_arch MATCHES "^([0-9]+[a-z]?)-real$")
            list(APPEND _gencode "-gencode=arch=compute_${CMAKE_MATCH_1},code=sm_${CMAKE_MATCH_1}")
        elseif(_arch MATCHES "^([0-9]+[a-z]?)-virtual$")
            list(APPEND _gencode
                 "-gencode=arch=compute_${CMAKE_MATCH_1},code=compute_${CMAKE_MATCH_1}")
        elseif(_arch MATCHES "^([0-9]+[a-z]?)$")
            list(APPEND _gencode
                 "-gencode=arch=compute_${CMAKE_MATCH_1},code=[sm_${CMAKE_MATCH_1},compute_${CMAKE_MATCH_1}]")
        else()
            message(FATAL_ERROR
                "DIFF_SOUP_CUDA_ARCHITECTURES entries are NN, NN-real or NN-virtual; got '${_arch}'.")
        endif()
    endforeach()
    set(_objects "")
    foreach(_name status_kernels assembly_kernels)
        set(_source "${CMAKE_CURRENT_SOURCE_DIR}/native/${_name}.cu")
        set(_object "${CMAKE_CURRENT_BINARY_DIR}/${_name}_cu${LINE}${CMAKE_CXX_OUTPUT_EXTENSION}")
        add_custom_command(
            OUTPUT "${_object}"
            COMMAND "${_nvcc}" -c "${_source}" -o "${_object}" -ccbin "${CMAKE_CXX_COMPILER}"
                    ${_host_flags} -std=c++17 -O3 -Xcompiler=-fPIC,-fvisibility=hidden
                    ${_gencode} "-I${CU${LINE}_CUDART_INCLUDE}"
            DEPENDS "${_source}" "${CMAKE_CURRENT_SOURCE_DIR}/native/${_name}.h"
            COMMENT "Compiling the CUDA ${LINE} ${_name} with ${_nvcc}"
            VERBATIM
        )
        set_source_files_properties("${_object}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
        list(APPEND _objects "${_object}")
    endforeach()
    set(${OUT} "${_objects}" PARENT_SCOPE)
endfunction()

set(_lines_built "")
foreach(LINE ${_resolved_lines})

    set(_cudss_include "${CU${LINE}_CUDSS_INCLUDE}")
    set(_cudss_library "${CU${LINE}_CUDSS_LIBRARY}")
    if(NOT CUDSS_INCLUDE_OVERRIDE STREQUAL "")
        set(_cudss_include "${CUDSS_INCLUDE_OVERRIDE}")
        message(STATUS "cudss_ffi: cuDSS include overridden -> ${_cudss_include}")
    endif()
    if(NOT CUDSS_LIBRARY_OVERRIDE STREQUAL "")
        set(_cudss_library "${CUDSS_LIBRARY_OVERRIDE}")
        message(STATUS "cudss_ffi: cuDSS library overridden -> ${_cudss_library}")
    endif()

    # The cuDSS release of the headers: the extension is written against the
    # cuDSS 1.x API and stops here below 1.0. cudss.h spells the version as
    # CUDSS_VERSION_MAJOR/_MINOR/_PATCH; CUDSS_VER_* is read as well.
    get_filename_component(_cudss_wheel_dir "${_cudss_include}" DIRECTORY)
    set(_cudss_header "${_cudss_include}/cudss.h")
    if(NOT EXISTS "${_cudss_header}")
        message(FATAL_ERROR
            "cudss_ffi: the cuDSS include directory ${_cudss_include} holds no cudss.h; "
            "install nvidia-cudss-cu${LINE} 1.x in the build environment.")
    endif()
    file(STRINGS "${_cudss_header}" _cudss_version_lines
         REGEX "^[ \t]*#[ \t]*define[ \t]+CUDSS_VER(SION)?_(MAJOR|MINOR|PATCH)[ \t]+[0-9]+")
    set(_cudss_version_parts "")
    foreach(_part MAJOR MINOR PATCH)
        if("${_cudss_version_lines}" MATCHES "CUDSS_VER(SION)?_${_part}[ \t]+([0-9]+)")
            list(APPEND _cudss_version_parts "${CMAKE_MATCH_2}")
        else()
            message(FATAL_ERROR
                "cudss_ffi: ${_cudss_header} defines no CUDSS_VERSION_${_part}, so the cuDSS "
                "release of the wheel at ${_cudss_wheel_dir} is unknown; cuDSS 1.0 or newer is "
                "required (nvidia-cudss-cu${LINE} 1.x).")
        endif()
    endforeach()
    list(GET _cudss_version_parts 0 _cudss_major)
    list(JOIN _cudss_version_parts "." _cudss_version)
    if(_cudss_major LESS 1)
        message(FATAL_ERROR
            "cudss_ffi: cuDSS 1.0 or newer is required; the wheel at ${_cudss_wheel_dir} holds "
            "cuDSS ${_cudss_version}. Install nvidia-cudss-cu${LINE}>=1.0.0,<2 in the build "
            "environment (pip install \"diff-soup[cu${LINE}]\").")
    endif()
    message(STATUS "cudss_ffi: CUDA ${LINE} builds against cuDSS ${_cudss_version} (${_cudss_wheel_dir})")

    set(_target _native_solver_cu${LINE})
    set(_subpackage cu${LINE})

    nanobind_add_module(${_target} NOMINSIZE
        native/single_solve.cpp
    )
    # Every line's extension is imported as <sub-package>._native_solver, so
    # the Python module name (and its PyInit symbol) stays _native_solver.
    set_target_properties(${_target} PROPERTIES OUTPUT_NAME _native_solver)

    # The CUDA kernels of the line (the status scan and the value assembly), compiled by that
    # line's nvcc (see the kernel build above) and linked into the module like any other object.
    if(_kernel_build STREQUAL "language")
        add_library(${_target}_kernels OBJECT native/status_kernels.cu native/assembly_kernels.cu)
        set_target_properties(${_target}_kernels PROPERTIES
            CUDA_ARCHITECTURES "${DIFF_SOUP_CUDA_ARCHITECTURES}"
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            POSITION_INDEPENDENT_CODE ON
            CUDA_VISIBILITY_PRESET hidden
            CUDA_RUNTIME_LIBRARY None
        )
        target_include_directories(${_target}_kernels PRIVATE ${CU${LINE}_CUDART_INCLUDE})
        target_sources(${_target} PRIVATE $<TARGET_OBJECTS:${_target}_kernels>)
    else()
        _diff_soup_cuda_kernel_objects(${LINE} _kernel_objects)
        target_sources(${_target} PRIVATE ${_kernel_objects})
    endif()
    # The module links the line's shared CUDA runtime itself (below); no runtime library is
    # added for the CUDA objects.
    set_target_properties(${_target} PROPERTIES CUDA_RUNTIME_LIBRARY None)

    target_include_directories(${_target} PRIVATE
        ${XLA_DIR}
        ${_cudss_include}
        ${CU${LINE}_CUDART_INCLUDE}
        ${CU${LINE}_NVCC_INCLUDE}
    )

    target_link_libraries(${_target} PRIVATE
        "${_cudss_library}"
        "${CU${LINE}_CUDART_LIBRARY}"
    )

    # Link libstdc++ and libgcc statically so the wheel is self-contained on
    # hosts whose system libstdc++ is older than the build compiler's.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_options(${_target} PRIVATE -static-libstdc++ -static-libgcc)
    endif()

    # Record the baked library search path as DT_RUNPATH so LD_LIBRARY_PATH
    # and the loader's explicit preload can override it (DT_RPATH cannot be
    # overridden at run time and a build in a temporary environment bakes in
    # directories that no longer exist). RUNPATH covers only this module's
    # direct dependencies; the NVIDIA libraries carry their own $ORIGIN
    # entries and resolve each other.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_options(${_target} PRIVATE "-Wl,--enable-new-dtags")
    endif()

    get_filename_component(_cudss_libdir "${_cudss_library}" DIRECTORY)
    get_filename_component(_cudart_libdir "${CU${LINE}_CUDART_LIBRARY}" DIRECTORY)

    # The .so lands at site-packages/diff_soup/_internal/cudss_ffi/cuNN/, so
    # $ORIGIN/../../../../nvidia/<pkg>/lib resolves to site-packages/nvidia/<pkg>/lib.
    set(_install_rpath "")
    foreach(_rel ${CU${LINE}_WHEEL_RPATH})
        list(APPEND _install_rpath "\$ORIGIN/../../../../${_rel}")
    endforeach()
    list(JOIN _install_rpath ";" _install_rpath_cmake)
    list(JOIN _install_rpath ":" _install_rpath_elf)

    set_target_properties(${_target} PROPERTIES
        BUILD_RPATH "${_cudss_libdir};${_cudart_libdir}"
        INSTALL_RPATH "${_install_rpath_cmake}"
        BUILD_WITH_INSTALL_RPATH OFF
        INSTALL_RPATH_USE_LINK_PATH OFF
    )

    # In-tree developer build: copy the module next to the Python sources of
    # its sub-package so the checkout imports it without a wheel install.
    set(_runtime_dst "${CMAKE_CURRENT_SOURCE_DIR}/${_subpackage}")
    add_custom_command(TARGET ${_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_runtime_dst}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:${_target}>"
                "${_runtime_dst}/$<TARGET_FILE_NAME:${_target}>"
        COMMENT "Copying CUDA ${LINE} _native_solver into diff_soup/_internal/cudss_ffi/${_subpackage}"
    )

    # install() rule consumed by scikit-build-core to place the module into
    # the wheel at diff_soup/_internal/cudss_ffi/cuNN/.
    install(TARGETS ${_target}
        LIBRARY DESTINATION diff_soup/_internal/cudss_ffi/${_subpackage}
    )

    # Force the installed RUNPATH to exactly the $ORIGIN-relative entries.
    # CMake's install-time edit rewrites only the part of the tag it generated;
    # absolute directories injected by conda or spack compiler wrappers survive
    # into the wheel, where they either do not exist or shadow the wheel's
    # libraries. patchelf replaces the whole tag. SKBUILD is defined only when
    # scikit-build-core drives the build, which is also where patchelf is
    # guaranteed present ([build-system].requires).
    if(DEFINED SKBUILD)
        find_program(PATCHELF_EXE patchelf)
        if(NOT PATCHELF_EXE)
            message(FATAL_ERROR
                "cudss_ffi: patchelf not found. It is required to normalise the "
                "RUNPATH of _native_solver in the wheel; add 'patchelf' to "
                "[build-system].requires.")
        endif()
        install(CODE "
            file(GLOB _cudss_so_glob \"\${CMAKE_INSTALL_PREFIX}/diff_soup/_internal/cudss_ffi/${_subpackage}/_native_solver*.so\")
            foreach(_so \${_cudss_so_glob})
                message(STATUS \"cudss_ffi: setting RUNPATH=${_install_rpath_elf} on \${_so}\")
                execute_process(
                    COMMAND \"${PATCHELF_EXE}\" --set-rpath \"${_install_rpath_elf}\" \"\${_so}\"
                    RESULT_VARIABLE _pe_rc)
                if(NOT _pe_rc EQUAL 0)
                    message(FATAL_ERROR \"cudss_ffi: patchelf failed on \${_so} (rc=\${_pe_rc})\")
                endif()
            endforeach()
        ")
    endif()

    list(APPEND _lines_built ${LINE})
endforeach()

if(NOT _lines_built)
    message(FATAL_ERROR
        "cudss_ffi: no CUDA line could be built. Install the NVIDIA wheels for "
        "CUDA 12 (nvidia-cudss-cu12 1.x, nvidia-cuda-runtime-cu12, "
        "nvidia-cuda-nvcc-cu12) or CUDA 13 (nvidia-cudss-cu13 1.x, nvidia-cuda-runtime, "
        "nvidia-cuda-nvcc), or configure with -DDIFF_SOUP_BUILD_CUDSS=OFF.")
endif()
message(STATUS "cudss_ffi: building the cuDSS extension for CUDA line(s): ${_lines_built}")
