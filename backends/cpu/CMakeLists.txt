file(GLOB_RECURSE KERNEL_SOURCES "kernels/*.cpp")
file(GLOB_RECURSE DEVICE_SOURCES "device/*.cpp")
file(GLOB_RECURSE REGISTRY_SOURCES "registry/*.cpp")

if(INSIGHT_DYNAMIC_CPU_BACKEND)
    add_library(insight_cpu_backend SHARED
        ${KERNEL_SOURCES}
        ${DEVICE_SOURCES}
        ${REGISTRY_SOURCES}
    )
    target_compile_definitions(insight_cpu_backend PRIVATE INSIGHT_CPU_BACKEND_EXPORTS)
    # DLL needs to link against insight_core for C API symbols
    target_link_libraries(insight_cpu_backend PRIVATE insight_core)
else()
    add_library(insight_cpu_backend STATIC
        ${KERNEL_SOURCES}
        ${DEVICE_SOURCES}
        ${REGISTRY_SOURCES}
    )
endif()

target_include_directories(insight_cpu_backend
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

if(INSIGHT_USE_FFTW3)
    find_package(FFTW3)
    if(FFTW3_FOUND)
        target_link_libraries(insight_cpu_backend PRIVATE ${FFTW3_LIBRARIES})
        target_compile_definitions(insight_cpu_backend PUBLIC INSIGHT_USE_FFTW3)
        message(STATUS "FFTW3 found and enabled for CPU backend")
    else()
        message(WARNING "FFTW3 not found, disabling FFT support")
        set(INSIGHT_USE_FFTW3 OFF)
    endif()
endif()

# OpenBLAS (optional, for accelerated linear algebra)
option(INSIGHT_USE_OPENBLAS "Use OpenBLAS for BLAS/LAPACK operations" ON)
if(INSIGHT_USE_OPENBLAS)
    find_package(OpenBLAS)
    if(OpenBLAS_FOUND)
        target_link_libraries(insight_cpu_backend PRIVATE ${OpenBLAS_LIBRARIES})
        target_compile_definitions(insight_cpu_backend PUBLIC INSIGHT_USE_OPENBLAS)
        message(STATUS "OpenBLAS found and enabled for CPU backend")
    else()
        message(WARNING "OpenBLAS not found, falling back to CLAPACK for linear algebra")
        set(INSIGHT_USE_OPENBLAS OFF)
    endif()
else()
    message(STATUS "OpenBLAS disabled by user option")
endif()