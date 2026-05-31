---
name: fetchcontent-xiph-codecs
description: Building libsndfile with Ogg/Vorbis/FLAC/Opus — now uses local third_party/ sources (see migrate-to-local-third-party skill)
source: auto-skill
extracted_at: '2026-05-29T19:31:53.038Z'
---

# Building libsndfile with Full Codec Support

## Status: Migrated to local third_party/ (2026-05-30)

As of 2026-05-30, all Xiph codec libraries are cloned to `third_party/`:
- `third_party/ogg` (v1.3.5)
- `third_party/vorbis` (v1.3.7)
- `third_party/flac` (1.4.3)
- `third_party/opus` (v1.5.2)
- `third_party/libsndfile` (1.2.2)

CMakeLists.txt auto-detects local sources and falls back to FetchContent if missing.
See `migrate-to-local-third-party` skill for the general pattern.

## Key Pitfalls (still relevant)

### 1. Cache variables must be set BEFORE FetchContent_MakeAvailable
libsndfile's `cmake/FindOgg.cmake`, `FindFLAC.cmake`, `FindVorbis.cmake`, `FindOpus.cmake` use `find_path`/`find_library`. If cache variables are set AFTER `FetchContent_MakeAvailable`, the find modules run during `MakeAvailable` and fail because the libraries haven't been built yet.

**Fix:** Use FetchContent naming convention (`_deps/<name>-src`, `_deps/<name>-build`) to pre-compute paths:

```cmake
set(_fc_base "${CMAKE_BINARY_DIR}/_deps")
set(OGG_INCLUDE_DIR "${_fc_base}/ogg-src/include" CACHE PATH "" FORCE)
set(OGG_LIBRARY "${_fc_base}/ogg-build/${CMAKE_STATIC_LIBRARY_PREFIX}ogg${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE FILEPATH "" FORCE)
# ... same for FLAC, Vorbis (vorbis, vorbisenc, vorbisfile), Opus
```

### 2. Use ALIAS targets for add_subdirectory deps (NOT IMPORTED)

When codec deps are built via `add_subdirectory`, the real targets (`vorbis`, `vorbisenc`, `FLAC`) exist but the namespaced targets (`Vorbis::vorbisenc`, `FLAC::FLAC`) do NOT (vorbis and FLAC don't create aliases; ogg and opus do).

Creating IMPORTED targets with file paths as `IMPORTED_LOCATION` breaks the dependency chain — CMake treats the file path as a literal dependency without a build rule, causing `No rule to make target '_deps/vorbis-build/lib/libvorbisenc.a'`.

**Fix**: Create ALIAS targets when the base target exists:
```cmake
foreach(_spec "Vorbis::vorbisenc|..." "FLAC::FLAC|..." ...)
    string(REPLACE "::" ";" _name_parts "${_target}")
    list(GET _name_parts 1 _base_name)
    if(TARGET ${_base_name})
        add_library(${_target} ALIAS ${_base_name})  # ← links to real target
    elseif(NOT TARGET ${_target})
        add_library(${_target} STATIC IMPORTED)       # ← fallback for FetchContent
        set_target_properties(${_target} PROPERTIES ...)
    endif()
endforeach()
```

**Which deps create their own aliases** (as of current versions):
- `ogg` → creates `Ogg::ogg` ALIAS ✅
- `opus` → creates `Opus::opus` ALIAS ✅
- `vorbis` → does NOT create `Vorbis::*` aliases ❌
- `FLAC` → does NOT create `FLAC::FLAC` alias ❌

### 3. `_resolve_dep` macro: save/restore in BOTH branches

The macro has `if(local)/else(FetchContent)` branches. Both must save `CMAKE_SKIP_INSTALL_RULES` before modifying it. Common bug: only the `if` branch saves `_saved_skip`, leaving the `else` branch to restore a stale or undefined value.

**Fix**: Move save before the `if`:
```cmake
macro(_resolve_dep VAR NAME)
    set(_saved_skip ${CMAKE_SKIP_INSTALL_RULES})    # ← save ONCE, before both branches
    if(DEFINED ${VAR})
        set(CMAKE_SKIP_INSTALL_RULES ON)
        add_subdirectory(...)
        set(CMAKE_SKIP_INSTALL_RULES ${_saved_skip})
    else()
        set(CMAKE_SKIP_INSTALL_RULES ON)
        FetchContent_MakeAvailable(...)
        set(CMAKE_SKIP_INSTALL_RULES ${_saved_skip})
    endif()
endmacro()
```

### 4. CMAKE_SKIP_INSTALL_RULES to avoid export conflicts
`FetchContent_MakeAvailable` triggers `install(EXPORT)` rules that fail when targets span multiple subdirectories (e.g., FLAC export depends on ogg not being in any export set).

**Fix:** Wrap the MakeAvailable call:
```cmake
set(_saved ${CMAKE_SKIP_INSTALL_RULES})
set(CMAKE_SKIP_INSTALL_RULES ON)
FetchContent_MakeAvailable(ogg vorbis FLAC opus)
set(CMAKE_SKIP_INSTALL_RULES ${_saved})
```

**⚠️ Also apply to `_resolve_dep` macro's FetchContent path**: The `_resolve_dep` helper macro has two branches — `add_subdirectory` (already protected) and `FetchContent_MakeAvailable` (must add protection). Both paths need `CMAKE_SKIP_INSTALL_RULES=ON`.

### 5. Ogg needs binary dir for generated config_types.h
`ogg/config_types.h` is generated during build, lives in `${ogg_BINARY_DIR}/include/`, not in the source tree.

**Fix:**
```cmake
set_target_properties(Ogg::ogg PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_fc_base}/ogg-src/include;${_fc_base}/ogg-build/include")
```

### 6. Opus header directory mismatch
Opus puts headers directly in `include/opus.h`, but code expects `include/opus/opus.h` (standard install layout).

**Fix:** Create a symbolic link:
```cmake
if(NOT EXISTS "${_fc_base}/opus-src/include/opus")
    file(CREATE_LINK "${_fc_base}/opus-src/include" "${_fc_base}/opus-src/include/opus" SYMBOLIC)
endif()
```

### 7. HAVE_EXTERNAL_XIPH_LIBS is unconditional
Line 127 of libsndfile's CMakeLists.txt: `set(HAVE_EXTERNAL_XIPH_LIBS ${ENABLE_EXTERNAL_LIBS})`. This means ALL xiph libs (including Opus) are linked, not just the ones found. You must provide ALL of them.

### 8. Proxy for AI Studio
Git fetches may hang without proxy:
```cmake
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
```

## Complete CMakeLists.txt Pattern

```cmake
option(INSIGHT_USE_SNDFILE "Use libsndfile for audio file I/O" ON)
if(INSIGHT_USE_SNDFILE)
    find_package(LibSndFile QUIET)  # Try system first
    if(NOT SndFile_FOUND)
        include(FetchContent)
        # Declare all deps
        FetchContent_Declare(ogg GIT_REPOSITORY "https://github.com/xiph/ogg.git" GIT_TAG "v1.3.5" GIT_SHALLOW TRUE)
        FetchContent_Declare(vorbis GIT_REPOSITORY "https://github.com/xiph/vorbis.git" GIT_TAG "v1.3.7" GIT_SHALLOW TRUE)
        FetchContent_Declare(FLAC GIT_REPOSITORY "https://github.com/xiph/flac.git" GIT_TAG "1.4.3" GIT_SHALLOW TRUE)
        FetchContent_Declare(opus GIT_REPOSITORY "https://github.com/xiph/opus.git" GIT_TAG "v1.5.2" GIT_SHALLOW TRUE)
        # Disable unnecessary features
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
        set(BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
        set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(INSTALL_MANPAGES OFF CACHE BOOL "" FORCE)
        set(INSTALL_PKGCONFIG_MODULES OFF CACHE BOOL "" FORCE)
        set(INSTALL_CMAKE_PACKAGE_MODULE OFF CACHE BOOL "" FORCE)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        set(BUILD_CXXLIBS OFF CACHE BOOL "" FORCE)
        # Pre-set cache vars using FetchContent naming convention
        set(_fc_base "${CMAKE_BINARY_DIR}/_deps")
        # ... set all OGG_, FLAC_, Vorbis_, OPUS_ cache vars ...
        # ... create imported targets with | separator ...
        # ... set INTERFACE_LINK_LIBRARIES for deps ...
        # ... fix Ogg include dir for config_types.h ...
        # Fetch and build
        set(_saved ${CMAKE_SKIP_INSTALL_RULES})
        set(CMAKE_SKIP_INSTALL_RULES ON)
        FetchContent_MakeAvailable(ogg vorbis FLAC opus)
        set(CMAKE_SKIP_INSTALL_RULES ${_saved})
        # Fix Opus header symlink
        # Fetch libsndfile
        FetchContent_Declare(sndfile ...)
        set(ENABLE_EXTERNAL_LIBS ON CACHE BOOL "" FORCE)
        set(HAVE_EXTERNAL_XIPH_LIBS ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(sndfile)
    endif()
    # Link
    target_link_libraries(insight_core PRIVATE SndFile::sndfile-static)
    target_compile_definitions(insight_core PUBLIC INSIGHT_USE_SNDFILE)
endif()
```
