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

# Copy ALL backend .so/.dll files to package directory for auto-discovery
# Cross-platform: Linux/macOS use libinsight_*_backend.so, Windows uses insight_*_backend.dll
file(GLOB _INSIGHT_BACKEND_SO_FILES
    "${CMAKE_BINARY_DIR}/backends/*/libinsight_*_backend.so"
    "${CMAKE_BINARY_DIR}/backends/*/libinsight_*_backend.dylib"
    "${CMAKE_BINARY_DIR}/backends/*/insight_*_backend.dll"
)
foreach(_backend_so ${_INSIGHT_BACKEND_SO_FILES})
    get_filename_component(_backend_name ${_backend_so} NAME)
    add_custom_command(TARGET insight_python POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_backend_so}"
            "${CMAKE_SOURCE_DIR}/bindings/python/insight/${_backend_name}"
        COMMENT "Copying ${_backend_name} to Python package"
    )
endforeach()

message(STATUS "Python bindings enabled (pybind11, Python ${Python3_VERSION})")
