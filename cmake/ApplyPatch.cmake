# cmake/ApplyPatch.cmake
# Idempotent patch application for third-party dependencies.
# Usage: apply_patch(<source_dir> <patch_file>)
#
# Tries multiple methods in order:
#   1. git apply (best for git repos)
#   2. patch -p1 (fallback, works without git)
#   3. If both fail, reports error (does NOT write stamp)
#
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

    # Method 1: git apply (preferred for git repos)
    execute_process(
        COMMAND git apply --check "${PATCH_FILE}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE GIT_CHECK
        OUTPUT_QUIET ERROR_QUIET
    )
    if(GIT_CHECK EQUAL 0)
        execute_process(
            COMMAND git apply "${PATCH_FILE}"
            WORKING_DIRECTORY "${SOURCE_DIR}"
            RESULT_VARIABLE GIT_APPLY
            ERROR_VARIABLE GIT_ERROR
        )
        if(GIT_APPLY EQUAL 0)
            file(WRITE "${PATCH_STAMP}" "Applied via git apply")
            message(STATUS "Patch ${PATCH_NAME} applied successfully (git apply)")
            return()
        endif()
    endif()

    # Method 2: patch -p1 (fallback)
    execute_process(
        COMMAND patch -p1 --forward --dry-run
        INPUT_FILE "${PATCH_FILE}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE PATCH_CHECK
        OUTPUT_QUIET ERROR_QUIET
    )
    if(PATCH_CHECK EQUAL 0)
        execute_process(
            COMMAND patch -p1 --forward
            INPUT_FILE "${PATCH_FILE}"
            WORKING_DIRECTORY "${SOURCE_DIR}"
            RESULT_VARIABLE PATCH_APPLY
            ERROR_VARIABLE PATCH_ERROR
        )
        if(PATCH_APPLY EQUAL 0)
            file(WRITE "${PATCH_STAMP}" "Applied via patch -p1")
            message(STATUS "Patch ${PATCH_NAME} applied successfully (patch -p1)")
            return()
        endif()
    endif()

    # Method 3: Check if already applied (reverse dry-run)
    execute_process(
        COMMAND patch -p1 --forward -R --dry-run
        INPUT_FILE "${PATCH_FILE}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE REVERSE_CHECK
        OUTPUT_QUIET ERROR_QUIET
    )
    if(REVERSE_CHECK EQUAL 0)
        file(WRITE "${PATCH_STAMP}" "Already applied")
        message(STATUS "Patch ${PATCH_NAME} already applied (detected via reverse check)")
        return()
    endif()

    # All methods failed
    message(WARNING "Patch ${PATCH_NAME} FAILED to apply to ${SOURCE_DIR} — expect build errors!")
    # Do NOT write stamp — allow retry on next configure
endfunction()
