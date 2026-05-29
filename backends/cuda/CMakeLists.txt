# backends/cuda/CMakeLists.txt
enable_language(CUDA)

file(GLOB_RECURSE KERNEL_SOURCES "kernels/*.cu")
file(GLOB_RECURSE DEVICE_SOURCES "device/*.cpp")
file(GLOB_RECURSE REGISTRY_SOURCES "registry/*.cpp")

if(INSIGHT_DYNAMIC_GPU_BACKEND)
    add_library(insight_cuda_backend SHARED
        ${KERNEL_SOURCES}
        ${DEVICE_SOURCES}
        ${REGISTRY_SOURCES}
        init.cpp
    )
    target_compile_definitions(insight_cuda_backend PRIVATE INSIGHT_GPU_BACKEND_EXPORTS)
else()
    add_library(insight_cuda_backend STATIC
        ${KERNEL_SOURCES}
        ${DEVICE_SOURCES}
        ${REGISTRY_SOURCES}
        init.cpp
    )
endif()

target_include_directories(insight_cuda_backend
    PUBLIC ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(insight_cuda_backend
    PRIVATE CUDA::cudart CUDA::cufft CUDA::curand insight_core
)

if(INSIGHT_CUDA_USE_THRUST)
    target_compile_definitions(insight_cuda_backend PRIVATE INSIGHT_USE_THRUST)
    message(STATUS "CUDA backend: Thrust enabled")
else()
    message(STATUS "CUDA backend: Thrust disabled (pure CUDA fallback)")
endif()