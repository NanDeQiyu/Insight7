---
name: apply-third-party-patch
description: CMake mechanism for idempotent, reproducible patches to third-party dependencies (inspired by PaddlePaddle)
source: auto-skill
extracted_at: '2026-05-30T17:30:00.000Z'
---

# Apply Third-Party Patches

## Problem

Third-party libraries fetched via FetchContent or cloned locally may have
GCC/Clang compatibility bugs. Patches must be:
1. **Idempotent** — safe to run cmake multiple times
2. **Reproducible** — patch files stored in version control
3. **Automatic** — applied during cmake configure step

## Solution

### 1. Create patch file from git diff

```bash
cd third_party/matplotplusplus
# Make changes to fix compilation errors
git diff > patches/matplotplusplus/gcc-compat.patch
```

### 2. Create `cmake/ApplyPatch.cmake`

```cmake
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
        OUTPUT_QUIET ERROR_QUIET
    )

    if(CHECK_RESULT EQUAL 0)
        execute_process(
            COMMAND git apply "${PATCH_FILE}"
            WORKING_DIRECTORY "${SOURCE_DIR}"
            RESULT_VARIABLE APPLY_RESULT
        )
        if(APPLY_RESULT EQUAL 0)
            file(WRITE "${PATCH_STAMP}" "Applied")
            message(STATUS "Patch ${PATCH_NAME} applied successfully")
        else()
            message(WARNING "Failed to apply patch ${PATCH_NAME}")
        endif()
    else()
        message(STATUS "Patch ${PATCH_NAME} check failed (may already be applied)")
        file(WRITE "${PATCH_STAMP}" "Skipped")
    endif()
endfunction()
```

### 3. Use in CMakeLists.txt

```cmake
include(cmake/ApplyPatch.cmake)

# After add_subdirectory or FetchContent_MakeAvailable:
apply_patch("${LOCAL_MATPLOT}"
    "${CMAKE_CURRENT_SOURCE_DIR}/patches/matplotplusplus/gcc-compat.patch")
```

### 4. Directory structure

```
Insight7/
├── cmake/
│   └── ApplyPatch.cmake
├── patches/
│   └── matplotplusplus/
│       └── gcc-compat.patch
└── third_party/
    └── matplotplusplus/   (cloned or fetched)
```

## Key Properties

- **Stamp file**: `.patch_applied_<name>` in source dir prevents re-application
- **`git apply --check`**: Tests if patch applies cleanly before committing
- **Version controlled**: Patch files in `patches/` are tracked in git
- **Works with FetchContent**: Patch is applied after source is available

## How to Apply

When a third-party library has a build error:
1. Clone the library locally to `third_party/`
2. Fix the error, run `git diff` to generate patch
3. Save to `patches/<library>/<fix-name>.patch`
4. Add `apply_patch()` call in CMakeLists.txt
5. Verify: `cmake ..` shows "applied successfully"
6. Verify idempotency: second `cmake ..` shows "already applied"
7. Verify clean: reset library (`git checkout -- .`) and re-run cmake
