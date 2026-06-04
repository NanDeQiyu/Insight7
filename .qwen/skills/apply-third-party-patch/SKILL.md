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

### 2. Create `cmake/ApplyPatch.cmake` — 3 fallback methods

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

    # Method 1: git apply (preferred for git repos)
    execute_process(
        COMMAND git apply --check "${PATCH_FILE}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE GIT_CHECK OUTPUT_QUIET ERROR_QUIET
    )
    if(GIT_CHECK EQUAL 0)
        execute_process(
            COMMAND git apply "${PATCH_FILE}"
            WORKING_DIRECTORY "${SOURCE_DIR}"
            RESULT_VARIABLE GIT_APPLY ERROR_VARIABLE GIT_ERROR
        )
        if(GIT_APPLY EQUAL 0)
            file(WRITE "${PATCH_STAMP}" "Applied via git apply")
            message(STATUS "Patch ${PATCH_NAME} applied successfully (git apply)")
            return()
        endif()
    endif()

    # Method 2: patch -p1 (fallback, works without git)
    execute_process(
        COMMAND patch -p1 --forward --dry-run
        INPUT_FILE "${PATCH_FILE}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE PATCH_CHECK OUTPUT_QUIET ERROR_QUIET
    )
    if(PATCH_CHECK EQUAL 0)
        execute_process(
            COMMAND patch -p1 --forward
            INPUT_FILE "${PATCH_FILE}"
            WORKING_DIRECTORY "${SOURCE_DIR}"
            RESULT_VARIABLE PATCH_APPLY ERROR_VARIABLE PATCH_ERROR
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
        RESULT_VARIABLE REVERSE_CHECK OUTPUT_QUIET ERROR_QUIET
    )
    if(REVERSE_CHECK EQUAL 0)
        file(WRITE "${PATCH_STAMP}" "Already applied")
        message(STATUS "Patch ${PATCH_NAME} already applied (detected via reverse check)")
        return()
    endif()

    # All methods failed
    message(WARNING "Patch ${PATCH_NAME} FAILED to apply — expect build errors!")
    # Do NOT write stamp — allow retry on next configure
endfunction()
```

**Why 3 methods**: `git apply` fails on some FetchContent shallow clones.
`patch -p1` works without git. Reverse check detects already-applied patches.

### 3. Use in CMakeLists.txt — PaddlePaddle style

**CRITICAL**: Patches must be applied BEFORE `add_subdirectory`, not after.
This matches PaddlePaddle's `ExternalProject_Add` + `PATCH_COMMAND` pattern.

```cmake
include(cmake/ApplyPatch.cmake)

# Inside the local/FetchContent branches — NOT after endif()!
if(EXISTS "${LOCAL_MATPLOT}/CMakeLists.txt")
    # Local: patch FIRST, then add_subdirectory
    apply_patch("${LOCAL_MATPLOT}"
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/matplotplusplus/gcc-compat.patch")
    apply_patch("${LOCAL_MATPLOT}"
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/matplotplusplus/gnuplot-unknown-terminal.patch")
    add_subdirectory("${LOCAL_MATPLOT}" "${CMAKE_CURRENT_BINARY_DIR}/matplotplusplus")
else()
    # FetchContent: Populate → patch → add_subdirectory (NOT MakeAvailable!)
    FetchContent_Declare(matplotplusplus
        GIT_REPOSITORY "https://github.com/alandefreitas/matplotplusplus"
        GIT_TAG        "v1.2.1"
        GIT_SHALLOW    TRUE
    )
    set(MATPLOTPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Populate(matplotplusplus)          # Step 1: Download only
    apply_patch("${matplotplusplus_SOURCE_DIR}" ... # Step 2: Apply patches
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/matplotplusplus/gcc-compat.patch")
    add_subdirectory("${matplotplusplus_SOURCE_DIR}" # Step 3: Process patched source
        "${matplotplusplus_BINARY_DIR}")
endif()
```

> **Why NOT `FetchContent_MakeAvailable`**: It calls `add_subdirectory` immediately,
> registering targets BEFORE patches are applied. The patches modify source files
> but cmake's internal state may cache the unpatched timestamps.
>
> **`FetchContent_Populate`** downloads source only. You then apply patches and
> call `add_subdirectory` manually — same as PaddlePaddle's `PATCH_COMMAND`.

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

## CRITICAL: Never edit patch files by hand

Patch files have strict format requirements (hunk headers with exact line counts,
context lines with correct indentation). Editing by hand almost always produces
a corrupt patch that `git apply` rejects with `corrupt patch at line N`.

**Correct way to update a patch**:
1. Reset source to original: `cd third_party/<lib> && git checkout -- <file>`
2. Apply changes to the source file directly (edit the .cpp/.h)
3. Generate patch: `cd third_party/<lib> && git diff -- <file> > patches/<lib>/<name>.patch`
4. Verify: `git apply --check patches/<lib>/<name>.patch` must print nothing
5. Verify reverse: `cd third_party/<lib> && git checkout -- <file> && patch -p1 --dry-run < patches/<lib>/<name>.patch`

**Symptom of corrupt patch**: `git apply --check` prints `error: corrupt patch at line N`.
The patch silently fails in CI (stamp is written, error is swallowed).

## ApplyPatch.cmake must NOT write stamp on failure

The stamp mechanism must be:
- `git apply --check` succeeds → apply → write stamp
- `git apply --check` fails → try reverse check (`git apply --check -R`)
  - Reverse succeeds → patch already applied → write stamp
  - Reverse fails → **do NOT write stamp** (allow retry on next configure)
