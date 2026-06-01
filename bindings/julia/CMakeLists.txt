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

# Copy Julia module to build directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Insight.jl
    ${CMAKE_BINARY_DIR}/bindings/julia/Insight.jl
    COPYONLY
)

# Copy Julia module files (modules/ directory) to build directory
file(GLOB_RECURSE _julia_modules "${CMAKE_CURRENT_SOURCE_DIR}/modules/*.jl")
foreach(_jl ${_julia_modules})
    file(RELATIVE_PATH _rel ${CMAKE_CURRENT_SOURCE_DIR} ${_jl})
    configure_file(${_jl} ${CMAKE_BINARY_DIR}/bindings/julia/${_rel} COPYONLY)
endforeach()

message(STATUS "Julia bindings enabled (C ABI wrapper + Insight.jl)")
