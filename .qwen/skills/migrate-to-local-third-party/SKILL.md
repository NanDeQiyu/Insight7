---
name: migrate-to-local-third-party
description: Migrating FetchContent dependencies to local third_party/ with automatic fallback, handling case-sensitive names, ALIAS targets, and PRIVATE link propagation
source: auto-skill
extracted_at: '2026-05-30T18:00:00.000Z'
---

# Migrate FetchContent Dependencies to Local third_party/

## Problem

FetchContent downloads dependencies from GitHub on every clean build, which:
- Is slow (network latency, especially behind proxies)
- Can fail (network outages, rate limits)
- Is not reproducible (different commits on different days)

## Solution

Clone dependencies to `third_party/` once, then use `add_subdirectory()` with
automatic fallback to FetchContent if local sources are missing.

## Step 1: Clone dependencies

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

cd third_party
git clone --depth 1 --branch <tag> https://github.com/<org>/<repo>.git
```

## Step 2: Update CMakeLists.txt

### Pattern: Auto-detect local sources with case-insensitive matching

```cmake
set(_local "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

foreach(_dep ogg vorbis FLAC opus sndfile)
    string(TOUPPER "${_dep}" _dep_upper)
    string(TOLOWER "${_dep}" _dep_lower)
    # Try exact name, lowercase, and lib-prefixed
    if(EXISTS "${_local}/${_dep}/CMakeLists.txt")
        set(${_dep_upper}_LOCAL "${_local}/${_dep}")
    elseif(EXISTS "${_local}/${_dep_lower}/CMakeLists.txt")
        set(${_dep_upper}_LOCAL "${_local}/${_dep_lower}")
    elseif(EXISTS "${_local}/lib${_dep_lower}/CMakeLists.txt")
        set(${_dep_upper}_LOCAL "${_local}/lib${_dep_lower}")
    endif()
endforeach()
```

**Why case-insensitive:** Git repos may be named `FLAC` or `flac`, `libsndfile` or `sndfile`.

### Pattern: _resolve_dep macro

```cmake
macro(_resolve_dep VAR NAME)
    if(DEFINED ${VAR})
        set(_${NAME}_src "${${VAR}}")
        set(_saved_skip ${CMAKE_SKIP_INSTALL_RULES})
        set(CMAKE_SKIP_INSTALL_RULES ON)
        add_subdirectory("${${VAR}}" "${CMAKE_BINARY_DIR}/_deps/${NAME}-build")
        set(CMAKE_SKIP_INSTALL_RULES ${_saved_skip})
    else()
        FetchContent_MakeAvailable(${NAME})
        set(_${NAME}_src "${${NAME}_SOURCE_DIR}")
    endif()
endmacro()
```

**Key:** `CMAKE_SKIP_INSTALL_RULES ON` prevents install export conflicts.

## Step 3: Handle ALIAS targets

When `add_subdirectory` creates targets, they may be ALIAS targets.
`set_target_properties` fails on ALIAS targets.

**Fix:** Check if target is IMPORTED before setting properties:

```cmake
get_target_property(_flac_type FLAC::FLAC TYPE)
if(_flac_type STREQUAL "STATIC_LIBRARY")
    get_target_property(_flac_imp FLAC::FLAC IMPORTED)
    if(_flac_imp)
        # Only set properties on targets WE created (imported)
        set_target_properties(FLAC::FLAC PROPERTIES ...)
    endif()
endif()
```

## Step 4: Handle PRIVATE link propagation

When a library (e.g., libsndfile) links its dependencies as PRIVATE,
consumers don't get those transitive deps. Fix by explicitly linking:

```cmake
target_link_libraries(insight_core PRIVATE SndFile::sndfile)
# sndfile's codec deps are PRIVATE, must link explicitly
if(TARGET Vorbis::vorbisenc)
    target_link_libraries(insight_core PRIVATE
        Vorbis::vorbisenc Vorbis::vorbis Vorbis::vorbisfile)
endif()
if(TARGET FLAC::FLAC)
    target_link_libraries(insight_core PRIVATE FLAC::FLAC)
endif()
```

## Step 5: Pre-set find_package variables

When using `add_subdirectory`, downstream `find_package(Ogg)` inside vorbis
may not find the ogg target. Pre-set the cache variables:

```cmake
set(OGG_INCLUDE_DIR "${_local}/ogg/include" CACHE PATH "" FORCE)
set(OGG_LIBRARY "${CMAKE_BINARY_DIR}/_deps/ogg-build/libogg.a" CACHE FILEPATH "" FORCE)
set(Ogg_FOUND TRUE CACHE BOOL "" FORCE)
set(CMAKE_MODULE_PATH "${_local}/vorbis/cmake;${CMAKE_MODULE_PATH}")
```

## Step 6: Apply patches if needed

Use the `apply-third-party-patch` skill for GCC/Clang compatibility fixes.

## Verification

```bash
# Clean build
rm -rf build/_deps build/CMakeCache.txt
cmake .. 2>&1 | grep "using local"  # Should show all deps
# Expected output:
#   ogg: using local .../third_party/ogg
#   vorbis: using local .../third_party/vorbis
#   FLAC: using local .../third_party/flac
#   opus: using local .../third_party/opus
#   sndfile: using local .../third_party/libsndfile

# Verify idempotency
cmake .. 2>&1 | grep "using local"  # Same output
```

## Directory structure

```
Insight7/
├── third_party/
│   ├── ogg/           (git clone --depth 1 --branch v1.3.5)
│   ├── vorbis/        (git clone --depth 1 --branch v1.3.7)
│   ├── flac/          (git clone --depth 1 --branch 1.4.3)
│   ├── opus/          (git clone --depth 1 --branch v1.5.2)
│   ├── libsndfile/    (git clone --depth 1 --branch 1.2.2)
│   └── matplotplusplus/ (git clone --depth 1 --branch v1.2.1)
├── patches/
│   └── matplotplusplus/
│       └── gcc-compat.patch
└── cmake/
    └── ApplyPatch.cmake
```
