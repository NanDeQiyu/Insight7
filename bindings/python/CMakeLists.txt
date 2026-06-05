# bindings/python/CMakeLists.txt
#
# Python bindings via pybind11.

find_package(Python3 COMPONENTS Interpreter Development NumPy QUIET)
if(NOT Python3_FOUND)
    message(WARNING "Python3 development libraries not found, skipping Python bindings")
    return()
endif()

# pybind11 from third_party
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11
                 ${CMAKE_BINARY_DIR}/pybind11 EXCLUDE_FROM_ALL)

pybind11_add_module(insight_python insight_py.cpp)
set_target_properties(insight_python PROPERTIES
    OUTPUT_NAME "_insight"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bindings/python"
)

target_include_directories(insight_python PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(insight_python PRIVATE
    insight_core
)

# Ensure backend .so is built before POST_BUILD copy
add_dependencies(insight_python insight_cpu_backend)

# Copy the shared library for easy import
add_custom_command(TARGET insight_python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:insight_python>
        "${CMAKE_BINARY_DIR}/bindings/python/$<TARGET_FILE_NAME:insight_python>"
    # Also copy to the wrapper package directory so `import insight` works
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:insight_python>
        "${CMAKE_SOURCE_DIR}/bindings/python/insight/$<TARGET_FILE_NAME:insight_python>"
    COMMENT "Copying Python binding"
)

# Copy ALL backend .so/.dll files to package directory for auto-discovery.
# NOTE: file(GLOB) runs at configure time — if backends aren't built yet,
# the glob is empty and no copy commands are registered. Instead, use a
# CMake -P script that globs at BUILD time.
add_custom_command(TARGET insight_python POST_BUILD
    COMMAND ${CMAKE_COMMAND}
        -DSRC_DIR="${CMAKE_BINARY_DIR}/backends"
        -DDST_DIR="${CMAKE_SOURCE_DIR}/bindings/python/insight"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/copy_backends.cmake"
    COMMENT "Copying backend .so files to Python package"
)

message(STATUS "Python bindings enabled (pybind11, Python ${Python3_VERSION})")
