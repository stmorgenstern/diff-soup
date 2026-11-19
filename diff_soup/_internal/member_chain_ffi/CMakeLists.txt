# The member chain extension, _native_member_chain: block Thomas over many independent block-tridiagonal chains,
# threaded over chains with OpenMP (the processor path of ds.linear.MemberSchur, elimination="native"). It links
# nothing but the OpenMP runtime, which the installed module finds in the KLU extension's lib/ (that directory
# bundles libgomp), as the SuperLU_MT extension does. It is added from klu_ffi/CMakeLists.txt, so it is built
# whenever the KLU extension is.
find_package(OpenMP REQUIRED COMPONENTS CXX)

nanobind_add_module(_native_member_chain NOMINSIZE
    native/member_chain.cpp
)
target_include_directories(_native_member_chain PRIVATE ${XLA_DIR})
target_compile_options(_native_member_chain PRIVATE -O3)
target_link_libraries(_native_member_chain PRIVATE OpenMP::OpenMP_CXX)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(_native_member_chain PRIVATE -static-libstdc++ -static-libgcc)
endif()

# A checkout build finds the OpenMP runtime next to the build compiler's runtime; an installed module finds it in
# the KLU extension's lib/.
set(_mc_build_rpath "")
foreach(_omp_lib ${OpenMP_CXX_LIBRARIES})
    get_filename_component(_omp_dir "${_omp_lib}" DIRECTORY)
    list(APPEND _mc_build_rpath "${_omp_dir}")
endforeach()
list(REMOVE_DUPLICATES _mc_build_rpath)
set_target_properties(_native_member_chain PROPERTIES
    BUILD_RPATH "${_mc_build_rpath}"
    INSTALL_RPATH "\$ORIGIN/../klu_ffi/lib"
)

if(NOT DEFINED SKBUILD)
    add_custom_command(TARGET _native_member_chain POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:_native_member_chain>"
                "${CMAKE_CURRENT_SOURCE_DIR}/$<TARGET_FILE_NAME:_native_member_chain>"
        COMMENT "Copying _native_member_chain into diff_soup/_internal/member_chain_ffi"
    )
endif()

install(TARGETS _native_member_chain
    LIBRARY DESTINATION diff_soup/_internal/member_chain_ffi
)
