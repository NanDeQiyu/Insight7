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
        ERROR_QUIET
    )

    if(CHECK_RESULT EQUAL 0)
        execute_process(
            COMMAND git apply "${PATCH_FILE}"
            WORKING_DIRECTORY "${SOURCE_DIR}"
            RESULT_VARIABLE APPLY_RESULT
        )
        if(APPLY_RESULT EQUAL 0)
            file(WRITE "${PATCH_STAMP}" "Applied on ${CMAKE_CURRENT_LIST_LINE}")
            message(STATUS "Patch ${PATCH_NAME} applied successfully")
        else()
            message(WARNING "Failed to apply patch ${PATCH_NAME}")
        endif()
    else()
        # Patch may already be applied (e.g. from local clone)
        message(STATUS "Patch ${PATCH_NAME} check failed (may already be applied)")
        file(WRITE "${PATCH_STAMP}" "Skipped (already applied or incompatible)")
    endif()
endfunction()
