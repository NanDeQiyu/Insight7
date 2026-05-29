# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)

# ========== Set policies to suppress warnings ==========
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 NEW)
endif()
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()

# ========== Project definition without hardcoding CUDA ==========
# CUDA will be enabled later if available and requested
project(insight VERSION 1.0.0 LANGUAGES CXX)

# ========== Options ==========
option(INSIGHT_BUILD_TESTS "Build unit tests" ON)
option(INSIGHT_WITH_CUDA "Enable CUDA backend" ON)

option(INSIGHT_DYNAMIC_CPU_BACKEND "Build CPU backend as shared library (DLL/SO)" ON)
option(INSIGHT_DYNAMIC_GPU_BACKEND "Build GPU backend as shared library (DLL/SO)" ON)

# ========== CUDA language support (optional) ==========
if(INSIGHT_WITH_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "CUDA backend enabled: ${CMAKE_CUDA_COMPILER}")
    else()
        message(WARNING "CUDA compiler not found, disabling CUDA backend")
        set(INSIGHT_WITH_CUDA OFF)
    endif()
endif()

# ========== Global CUDA architecture settings ==========
if(INSIGHT_WITH_CUDA AND NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "52;61;70;75;80;86;89" CACHE STRING "CUDA architectures")
endif()

# ========== C++ standard ==========
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ========== Add custom CMake module path ==========
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ========== Include auto-fetch function ==========
include(FetchDependency)

# ========== Directory Definitions ==========
set(INSIGHT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(INSIGHT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INSIGHT_INTERNAL_DIR ${INSIGHT_SRC_DIR}/internal)
set(INSIGHT_UTILS_DIR ${INSIGHT_SRC_DIR}/utils)

# ========== Source Files Collection ==========
file(GLOB_RECURSE INSIGHT_SOURCES
    ${INSIGHT_SRC_DIR}/*/*.cpp
)

# ========== GoogleTest (for unit testing) ==========
if(INSIGHT_BUILD_TESTS)
    auto_fetch_dependency(googletest
        "https://github.com/google/googletest.git"
        "v1.15.0"
    )
endif()

# ========== Core Library ==========
add_library(insight_core STATIC
    ${INSIGHT_SOURCES}
    src/init.cpp
)
set_target_properties(insight_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ========== OpenMP ==========
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(insight_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(insight_core PRIVATE ${OpenMP_CXX_FLAGS})
    target_compile_definitions(insight_core PUBLIC INSIGHT_WITH_OPENMP)
endif()

# ========== Thrust ==========
option(INSIGHT_USE_THRUST "Use Thrust for GPU algorithms (sort, scan, etc.)" ON)

# ========== OpenBLAS ==========
option(INSIGHT_USE_OPENBLAS "Use OpenBLAS for linear algebra" ON)
if(INSIGHT_USE_OPENBLAS)
    find_package(OpenBLAS)
    if(OpenBLAS_FOUND)
        target_link_libraries(insight_core PRIVATE ${OpenBLAS_LIBRARIES})
        target_compile_definitions(insight_core PUBLIC INSIGHT_USE_OPENBLAS)
        message(STATUS "OpenBLAS found and enabled")
    else()
        message(WARNING "OpenBLAS not found, disabling INSIGHT_USE_OPENBLAS")
        set(INSIGHT_USE_OPENBLAS OFF)
    endif()
else()
    message(STATUS "OpenBLAS support disabled")
endif()

# ========== FFTW3 ==========
option(INSIGHT_USE_FFTW3 "Use FFTW3 for FFT operations" ON)
if(INSIGHT_USE_FFTW3)
    find_package(FFTW3)
    if(FFTW3_FOUND)
        target_link_libraries(insight_core PRIVATE ${FFTW3_LIBRARIES})
        target_compile_definitions(insight_core PUBLIC INSIGHT_USE_FFTW3)
        message(STATUS "FFTW3 found and enabled")
    else()
        message(WARNING "FFTW3 not found, disabling INSIGHT_USE_FFTW3")
        set(INSIGHT_USE_FFTW3 OFF)
    endif()
else()
    message(STATUS "FFTW3 support disabled")
endif()

# ========== Include Directories ==========
target_include_directories(insight_core
    PUBLIC
        ${INSIGHT_INCLUDE_DIR}
        ${OpenBLAS_INCLUDE_DIRS}
    PRIVATE
        ${INSIGHT_INTERNAL_DIR}
        ${INSIGHT_UTILS_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# ========== CPU Backend ==========
add_subdirectory(backends/cpu)
if(INSIGHT_DYNAMIC_CPU_BACKEND)
    message(STATUS "CPU backend: dynamic linkage (DLL/SO)")
    target_compile_definitions(insight_core PRIVATE INSIGHT_CPU_BACKEND_EXPORTS)
else()
    target_link_libraries(insight_core PRIVATE insight_cpu_backend)
    message(STATUS "CPU backend: static linkage")
endif()

# ========== CUDA Backend ==========
if(INSIGHT_WITH_CUDA)
    if(INSIGHT_USE_THRUST)
        set(INSIGHT_CUDA_USE_THRUST ON)
        target_compile_definitions(insight_core PUBLIC INSIGHT_USE_THRUST)
    else()
        set(INSIGHT_CUDA_USE_THRUST OFF)
    endif()
    add_subdirectory(backends/cuda)
    if(INSIGHT_DYNAMIC_GPU_BACKEND)
        message(STATUS "CUDA backend: dynamic linkage (DLL/SO)")
        target_compile_definitions(insight_core PRIVATE INSIGHT_GPU_BACKEND_EXPORTS)
    else()
        target_link_libraries(insight_core PRIVATE insight_cuda_backend)
        message(STATUS "CUDA backend: static linkage")
    endif()
    target_compile_definitions(insight_core PUBLIC INSIGHT_WITH_CUDA)
else()
    message(STATUS "CUDA backend disabled by user option")
endif()

# ========== Demos ==========
option(INSIGHT_BUILD_DEMOS "Build demo programs" ON)
if(INSIGHT_BUILD_DEMOS)
    add_subdirectory(demos)
endif()

# ========== Unit Tests ==========
if(INSIGHT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ========== Configuration Summary ==========
if(INSIGHT_DYNAMIC_CPU_BACKEND)
    set(CPU_BACKEND_TYPE "dynamic")
else()
    set(CPU_BACKEND_TYPE "static")
endif()

if(INSIGHT_DYNAMIC_GPU_BACKEND)
    set(GPU_BACKEND_TYPE "dynamic")
else()
    set(GPU_BACKEND_TYPE "static")
endif()

message(STATUS "")
message(STATUS "Insight Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "  OpenMP:    ${OpenMP_CXX_FOUND}")
message(STATUS "  OpenBLAS:  ${INSIGHT_USE_OPENBLAS}")
message(STATUS "  FFTW3:     ${INSIGHT_USE_FFTW3}")
message(STATUS "  CUDA:      ${INSIGHT_WITH_CUDA}")
message(STATUS "  Thrust:    ${INSIGHT_USE_THRUST}")
message(STATUS "  Tests:     ${INSIGHT_BUILD_TESTS}")
message(STATUS "  CPU backend: ${CPU_BACKEND_TYPE}")
message(STATUS "  GPU backend: ${GPU_BACKEND_TYPE}")
message(STATUS "")