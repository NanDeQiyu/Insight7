# bindings/julia/CMakeLists.txt
#
# Julia bindings: C ABI wrapper library (libinsight_julia.so).
# Julia loads this via ccall(). The Julia module is Insight.jl (pure Julia).

# Build C wrapper as shared library
add_library(insight_julia SHARED insight_julia_capi.cpp)
set_target_properties(insight_julia PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bindings/julia"
)

target_include_directories(insight_julia PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(insight_julia PRIVATE
    insight_core
)

# Ensure backend .so is built before POST_BUILD copy
add_dependencies(insight_julia insight_cpu_backend)

# Copy .so to source wrapper directory for Julia Pkg.develop
add_custom_command(TARGET insight_julia POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:insight_julia>
        "${CMAKE_SOURCE_DIR}/bindings/julia/$<TARGET_FILE_NAME:insight_julia>"
    # Copy CPU backend .so for dlopen
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_BINARY_DIR}/backends/cpu/libinsight_cpu_backend.so"
        "${CMAKE_SOURCE_DIR}/bindings/julia/libinsight_cpu_backend.so"
    COMMENT "Copying Julia binding + CPU backend"
)

# Copy Julia module to build directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Insight.jl
    ${CMAKE_BINARY_DIR}/bindings/julia/Insight.jl
    COPYONLY
)

# Copy Insight.jl to src/ for standard Julia package structure
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Insight.jl
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Insight.jl
    COPYONLY
)

# Copy modules/ to src/ for standard Julia package structure
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/modules
     DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Copy Julia module files (modules/ directory) to build directory
file(GLOB_RECURSE _julia_modules "${CMAKE_CURRENT_SOURCE_DIR}/modules/*.jl")
foreach(_jl ${_julia_modules})
    file(RELATIVE_PATH _rel ${CMAKE_CURRENT_SOURCE_DIR} ${_jl})
    configure_file(${_jl} ${CMAKE_BINARY_DIR}/bindings/julia/${_rel} COPYONLY)
endforeach()

# Also copy modules to source directory for Pkg.develop
add_custom_command(TARGET insight_julia POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/modules"
        "${CMAKE_SOURCE_DIR}/bindings/julia/modules"
    COMMENT "Copying Julia modules to source directory"
)

message(STATUS "Julia bindings enabled (C ABI wrapper + Insight.jl)")
