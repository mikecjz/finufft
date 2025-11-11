find_package(Matlab REQUIRED)
matlab_add_mex(NAME finufft_mex SRC finufft.cpp LINK_TO finufft OUTPUT_NAME finufft R2018a)
target_compile_definitions(finufft_mex PRIVATE -DR2008OO)

# Add CUDA MEX file if CUDA is enabled
if(FINUFFT_USE_CUDA)
    # Add MATLAB GPU Computing SDK include directory
    include_directories(${ _ROOT_DIR}/toolbox/parallel/gpu/extern/include)

    matlab_add_mex(NAME cufinufft_mex SRC cufinufft.cu LINK_TO cufinufft OUTPUT_NAME cufinufft R2018a)
    target_compile_definitions(cufinufft_mex PRIVATE -DR2008OO)
endif()

file(GLOB FINUFFT_MATLAB_M_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.m)

add_custom_command(
    TARGET finufft_mex
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${FINUFFT_MATLAB_M_SOURCES} ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
)

# Also copy .m files when building CUDA MEX
if(FINUFFT_USE_CUDA)
    add_custom_command(
        TARGET cufinufft_mex
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${FINUFFT_MATLAB_M_SOURCES} ${CMAKE_CURRENT_BINARY_DIR}
        VERBATIM
    )
endif()
