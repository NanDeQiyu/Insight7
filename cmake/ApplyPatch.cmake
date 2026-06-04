# cmake/ApplyPatch.cmake
# Idempotent patch application for third-party dependencies.
# Usage: apply_patch(<source_dir> <patch_file>)
#
# Applies a git patch to a source directory. Skips if already applied.
# Inspired by PaddlePaddle's patch mechanism.

function(apply_patch SOURCE_DIR PATCH_FILE)
    if(NOT EXISTS "${PATCH_FILE}")
        message(WARNING "Patch file not found: ${PATCH_FILE}")
        return()
    endif()

    get_filename_component(PATCH_NAME "${PATCH_FILE}" NAME_WE)
    set(PATCH_STAMP "${SOURCE_DIR}/.patch_applied_${PATCH_NAME}")

    if(EXISTS "${PATCH_STAMP}")
        message(STATUS "Patch ${PATCH_NAME} already applied to ${SOURCE_DIR}")
        return()
    endif()

    message(STATUS "Applying patch ${PATCH_NAME} to ${SOURCE_DIR}")
    execute_process(
        COMMAND git apply --check "${PATCH_FILE}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE CHECK_RESULT
        OUTPUT_QUIET
        ERROR_VARIABLE CHECK_ERROR
    )

    if(CHECK_RESULT EQUAL 0)
        execute_process(
            COMMAND git apply "${PATCH_FILE}"
            WORKING_DIRECTORY "${SOURCE_DIR}"
            RESULT_VARIABLE APPLY_RESULT
            ERROR_VARIABLE APPLY_ERROR
        )
        if(APPLY_RESULT EQUAL 0)
            file(WRITE "${PATCH_STAMP}" "Applied on ${CMAKE_CURRENT_LIST_LINE}")
            message(STATUS "Patch ${PATCH_NAME} applied successfully")
        else()
            message(WARNING "Failed to apply patch ${PATCH_NAME}: ${APPLY_ERROR}")
        endif()
    else()
        # Check if patch is already applied by trying reverse check
        execute_process(
            COMMAND git apply --check -R "${PATCH_FILE}"
            WORKING_DIRECTORY "${SOURCE_DIR}"
            RESULT_VARIABLE REVERSE_RESULT
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(REVERSE_RESULT EQUAL 0)
            file(WRITE "${PATCH_STAMP}" "Already applied")
            message(STATUS "Patch ${PATCH_NAME} already applied (detected via reverse check)")
        else()
            message(WARNING "Patch ${PATCH_NAME} cannot be applied and is not already applied: ${CHECK_ERROR}")
            # Do NOT write stamp — allow retry on next configure
        endif()
    endif()
endfunction()
