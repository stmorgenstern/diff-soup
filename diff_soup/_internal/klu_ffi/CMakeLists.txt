cmake_minimum_required(VERSION 3.18)
project(diff_soup_klu_ffi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The KLU extension, _native_klu, is a host sparse LU backend built on
# SuiteSparse KLU. It links the prebuilt SuiteSparse 7.10.1 shared libraries
# (libklu, libamd, libcolamd, libbtf, libsuitesparseconfig) and compiles
# against their headers. KLU_VENDOR_DIR names the tree that holds
# include/suitesparse/klu.h and lib/lib<name>.so.<N>; it defaults to the
# vendored SuiteSparse tree committed in the repository.
set(KLU_VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor" CACHE PATH
    "Tree holding include/suitesparse/klu.h and the SuiteSparse shared libraries in lib/")
get_filename_component(KLU_VENDOR_DIR "${KLU_VENDOR_DIR}" ABSOLUTE)
set(_KLU_INC "${KLU_VENDOR_DIR}/include/suitesparse")
set(_KLU_LIBDIR "${KLU_VENDOR_DIR}/lib")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" "-m" "nanobind" "--cmake_dir"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
    RESULT_VARIABLE NANOBIND_CMAKE_DIR_RESULT
)
if(NANOBIND_CMAKE_DIR_RESULT OR NOT NANOBIND_CMAKE_DIR)
    message(FATAL_ERROR "Failed to discover the nanobind CMake directory via 'python -m nanobind --cmake_dir'.")
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

if(NOT EXISTS "${_KLU_INC}/klu.h")
    message(FATAL_ERROR
        "klu_ffi: klu.h not found under KLU_VENDOR_DIR='${KLU_VENDOR_DIR}'. It needs "
        "include/suitesparse/klu.h and lib/libklu.so.<N>; pass -DKLU_VENDOR_DIR=<tree> naming a "
        "SuiteSparse 7 installation laid out that way.")
endif()

# The tree carries versioned sonames (libklu.so.2) and no unversioned
# development link, so each library is resolved by glob: the shortest match
# is the soname itself, which is what the module records as its dependency.
function(_klu_resolve_soname OUT_VAR STEM)
    file(GLOB _hits "${_KLU_LIBDIR}/lib${STEM}.so.[0-9]*")
    list(LENGTH _hits _count)
    if(_count EQUAL 0)
        message(FATAL_ERROR "klu_ffi: no lib${STEM}.so.<N> under ${_KLU_LIBDIR}.")
    endif()
    set(_best "")
    set(_best_dots 999)
    foreach(_hit ${_hits})
        get_filename_component(_name "${_hit}" NAME)
        string(REGEX REPLACE "^.*\\.so\\." "" _suffix "${_name}")
        string(REGEX MATCHALL "\\." _dots "${_suffix}")
        list(LENGTH _dots _ndots)
        if(_ndots LESS _best_dots)
            set(_best "${_hit}")
            set(_best_dots ${_ndots})
        endif()
    endforeach()
    set(${OUT_VAR} "${_best}" PARENT_SCOPE)
endfunction()

set(_KLU_STEMS klu amd colamd btf suitesparseconfig)
set(_KLU_LIBS "")
foreach(_stem ${_KLU_STEMS})
    _klu_resolve_soname(_lib ${_stem})
    list(APPEND _KLU_LIBS "${_lib}")
endforeach()
message(STATUS "klu_ffi: linking ${_KLU_LIBS}")

nanobind_add_module(_native_klu NOMINSIZE
    native/klu_ffi.cpp
)

target_include_directories(_native_klu PRIVATE
    ${XLA_DIR}
    ${_KLU_INC}
)

target_link_libraries(_native_klu PRIVATE ${_KLU_LIBS})
target_link_options(_native_klu PRIVATE "-Wl,-rpath-link,${_KLU_LIBDIR}")

# Link libstdc++ and libgcc statically so the module runs on hosts whose
# system libstdc++ is older than the build compiler's.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(_native_klu PRIVATE -static-libstdc++ -static-libgcc)
endif()

# A checkout build finds the SuiteSparse libraries in the vendored tree by
# absolute path; an installed module finds them in lib/ next to itself.
set_target_properties(_native_klu PROPERTIES
    BUILD_RPATH "${_KLU_LIBDIR}"
    INSTALL_RPATH "\$ORIGIN/lib"
    BUILD_WITH_INSTALL_RPATH OFF
    INSTALL_RPATH_USE_LINK_PATH OFF
)

# In-tree developer build: copy the module next to the Python sources so the
# checkout imports it without a wheel install. Skipped when scikit-build
# drives the build, where only the install() rules populate the wheel.
if(NOT DEFINED SKBUILD)
    add_custom_command(TARGET _native_klu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:_native_klu>"
                "${CMAKE_CURRENT_SOURCE_DIR}/$<TARGET_FILE_NAME:_native_klu>"
        COMMENT "Copying _native_klu into diff_soup/_internal/klu_ffi"
    )
endif()

# Installed layout:
#   diff_soup/_internal/klu_ffi/_native_klu*.so
#   diff_soup/_internal/klu_ffi/lib/   the SuiteSparse libraries and the GCC
#                                      runtime libraries they need, each with
#                                      its soname
#   diff_soup/_internal/klu_ffi/licenses/   the licence texts of the bundled libraries
#   diff_soup/_internal/klu_ffi/THIRD_PARTY_NOTICES.md
install(TARGETS _native_klu
    LIBRARY DESTINATION diff_soup/_internal/klu_ffi
)
set(_KLU_BUNDLE "")
foreach(_stem ${_KLU_STEMS} gomp gfortran quadmath gcc_s)
    file(GLOB _found "${_KLU_LIBDIR}/lib${_stem}.so.[0-9]*")
    list(APPEND _KLU_BUNDLE ${_found})
endforeach()
install(FILES ${_KLU_BUNDLE}
    DESTINATION diff_soup/_internal/klu_ffi/lib
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)
file(GLOB _KLU_LICENSES "${CMAKE_CURRENT_SOURCE_DIR}/licenses/*")
install(FILES ${_KLU_LICENSES}
    DESTINATION diff_soup/_internal/klu_ffi/licenses
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
    DESTINATION diff_soup/_internal/klu_ffi
)

# Conda compilers add an absolute -rpath to the active prefix through
# LDFLAGS; it would let a SuiteSparse in that prefix shadow the bundled one.
# The installed module's search path is rewritten to exactly $ORIGIN/lib
# (DT_RPATH, so it covers the bundled libraries' own dependencies too).
find_program(KLU_FFI_PATCHELF patchelf)
if(KLU_FFI_PATCHELF)
    install(CODE "
        file(GLOB _klu_so \"\${CMAKE_INSTALL_PREFIX}/diff_soup/_internal/klu_ffi/_native_klu*.so\")
        foreach(_so \${_klu_so})
            execute_process(
                COMMAND \"${KLU_FFI_PATCHELF}\" --force-rpath --set-rpath \"\\\$ORIGIN/lib\" \"\${_so}\"
                RESULT_VARIABLE _rc)
            if(NOT _rc EQUAL 0)
                message(FATAL_ERROR \"klu_ffi: patchelf could not set the search path of \${_so} (exit \${_rc}).\")
            endif()
        endforeach()
    ")
else()
    message(WARNING
        "klu_ffi: patchelf not found, so the installed _native_klu keeps the build's search "
        "path after \$ORIGIN/lib. Install patchelf (a pip wheel) for a self-contained install.")
endif()
