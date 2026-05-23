# tests/CMakeLists.txt

# ========== 收集测试源文件 ==========
file(GLOB CPU_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.cpp")
file(GLOB CUDA_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cpp")

add_executable(insight_tests_cpu ${CPU_TEST_SOURCES})
add_executable(insight_tests_cuda ${CUDA_TEST_SOURCES})

target_link_libraries(insight_tests_cpu
    insight_core
    insight_cpu_backend
    GTest::gtest_main
)

target_link_libraries(insight_tests_cuda
    insight_core
    insight_cpu_backend
    insight_cuda_backend
    GTest::gtest_main
)

target_include_directories(insight_tests_cpu
    PRIVATE
        ${INSIGHT_INCLUDE_DIR}
)

# ========== 复制运行时依赖 ==========

function(copy_runtime_dependencies TARGET_NAME BACKEND_TARGET)
    if(NOT BACKEND_TARGET)
        return()
    endif()
    
    add_dependencies(${TARGET_NAME} ${BACKEND_TARGET})
    
    # 复制后端 DLL
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:${BACKEND_TARGET}>"
            "$<TARGET_FILE_DIR:${TARGET_NAME}>/$<TARGET_FILE_NAME:${BACKEND_TARGET}>"
        COMMENT "Copying ${BACKEND_TARGET} to ${TARGET_NAME}"
    )
    
    # Windows: 复制后端目录下的所有 DLL
    if(WIN32)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${TARGET_NAME}>"
            COMMAND powershell -Command "Copy-Item -Path '$<TARGET_FILE_DIR:${BACKEND_TARGET}>/*.dll' -Destination '$<TARGET_FILE_DIR:${TARGET_NAME}>/' -Force -ErrorAction SilentlyContinue"
            COMMENT "Copying all DLL dependencies from backend directory"
        )
    endif()
endfunction()

# CPU 测试：只拷贝 CPU 后端
if(INSIGHT_DYNAMIC_CPU_BACKEND)
    copy_runtime_dependencies(insight_tests_cpu insight_cpu_backend)
endif()

# GPU 测试：拷贝 CPU + GPU 后端
if(INSIGHT_WITH_CUDA AND INSIGHT_DYNAMIC_GPU_BACKEND)
    copy_runtime_dependencies(insight_tests_cuda insight_cuda_backend)
    # GPU 测试也需要 CPU 后端（作为 fallback）
    copy_runtime_dependencies(insight_tests_cuda insight_cpu_backend)
endif()

# ========== 自动发现并注册测试 ==========
include(GoogleTest)
gtest_discover_tests(insight_tests_cpu)
if(INSIGHT_WITH_CUDA)
    gtest_discover_tests(insight_tests_cuda)
endif()