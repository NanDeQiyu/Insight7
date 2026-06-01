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

# Inside the local/FetchContent branches — NOT after endif()!
if(EXISTS "${LOCAL_MATPLOT}/CMakeLists.txt")
    add_subdirectory("${LOCAL_MATPLOT}" ...)
    apply_patch("${LOCAL_MATPLOT}"              # ← local path
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/matplotplusplus/gcc-compat.patch")
else()
    FetchContent_MakeAvailable(matplotplusplus)
    apply_patch("${matplotplusplus_SOURCE_DIR}"  # ← FetchContent _deps/ path
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/matplotplusplus/gcc-compat.patch")
endif()
```

> **⚠️ Critical**: `apply_patch` must be INSIDE each branch, not after `endif()`.
> `LOCAL_MATPLOT` always points to `third_party/`, but FetchContent downloads to `_deps/<name>-src/`.
> If called after `endif()` with only the LOCAL path, FetchContent builds silently skip the patch.

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

## ⚠️ Gotcha: .gitignore blocks cmake/ files

The project `.gitignore` has `*.cmake` which ignores ALL `.cmake` files in any directory. This silently prevents `cmake/ApplyPatch.cmake` (and other cmake modules) from being committed.

**Fix**: Add `!cmake/*.cmake` after `*.cmake` in `.gitignore`:
```
*.cmake
!CMakeLists.txt
!cmake/*.cmake
```

**How to detect**: `git check-ignore cmake/ApplyPatch.cmake` returns the filename (exit 0) if ignored.
**Symptom in CI**: `CMake Error at CMakeLists.txt:344 (include): include could not find requested file: cmake/ApplyPatch.cmake`

## ⚠️ Gotcha: apply_patch must use correct source dir per branch

When `CMakeLists.txt` has `if(local)/else(FetchContent)`, the source directory differs:
- Local: `third_party/<name>/` (the `LOCAL_<NAME>` variable)
- FetchContent: `_deps/<name>-src/` (the `<name>_SOURCE_DIR` variable)

If `apply_patch` is called after `endif()` with only the LOCAL path, FetchContent builds silently skip the patch (WARNING but no error).

**Fix**: Move `apply_patch` into each branch:
```cmake
if(local)
    add_subdirectory(...)
    apply_patch("${LOCAL_PATH}" ...)
else()
    FetchContent_MakeAvailable(...)
    apply_patch("${<name>_SOURCE_DIR}" ...)
endif()
```

**Symptom**: Template compilation errors in the patched file (e.g., `wrong number of template arguments`) even though the patch is committed. The patch was never applied to the FetchContent source.

## How to Apply

When a third-party library has a build error:
1. Clone the library locally to `third_party/`
2. Fix the error, run `git diff` to generate patch
3. Save to `patches/<library>/<fix-name>.patch`
4. Add `apply_patch()` call in CMakeLists.txt
5. **Verify `cmake/ApplyPatch.cmake` is tracked**: `git ls-files cmake/ApplyPatch.cmake`
6. Verify: `cmake ..` shows "applied successfully"
7. Verify idempotency: second `cmake ..` shows "already applied"
8. Verify clean: reset library (`git checkout -- .`) and re-run cmake
