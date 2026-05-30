# demos/CMakeLists.txt

file(GLOB DEMO_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Helper function to create a demo target with backend DLL copying
function(add_insight_demo TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})

    target_link_libraries(${TARGET_NAME} PRIVATE insight_core)

    target_include_directories(${TARGET_NAME}
        PRIVATE
            ${INSIGHT_INCLUDE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/..
    )

    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/demos
    )

    # Copy CPU backend DLL
    if(INSIGHT_DYNAMIC_CPU_BACKEND)
        add_dependencies(${TARGET_NAME} insight_cpu_backend)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:insight_cpu_backend>"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>/$<TARGET_FILE_NAME:insight_cpu_backend>"
            COMMENT "Copying CPU backend DLL to ${TARGET_NAME}"
        )
    endif()

    # Copy GPU backend DLL
    if(INSIGHT_WITH_CUDA AND INSIGHT_DYNAMIC_GPU_BACKEND)
        add_dependencies(${TARGET_NAME} insight_cuda_backend)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:insight_cuda_backend>"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>/$<TARGET_FILE_NAME:insight_cuda_backend>"
            COMMENT "Copying GPU backend DLL to ${TARGET_NAME}"
        )
    endif()
endfunction()

# Register each demo
add_insight_demo(demo_basic_ops   basic_ops.cpp)
add_insight_demo(demo_linalg      linalg_demo.cpp)
add_insight_demo(demo_fft         fft_demo.cpp)
add_insight_demo(demo_gpu_transfer gpu_transfer.cpp)

if(INSIGHT_USE_SNDFILE)
    add_insight_demo(demo_sndfile sndfile_demo.cpp)
    target_compile_definitions(demo_sndfile PRIVATE
        AUDIO_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../audio"
    )
endif()

add_insight_demo(demo_radar_task1 radar_task1.cpp)
