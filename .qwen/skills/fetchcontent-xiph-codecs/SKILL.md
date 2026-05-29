---
name: fetchcontent-xiph-codecs
description: Building libsndfile with Ogg/Vorbis/FLAC/Opus via FetchContent on systems without sudo
source: auto-skill
extracted_at: '2026-05-29T19:31:53.038Z'
---

# Building libsndfile with Full Codec Support via FetchContent

## Problem
libsndfile requires a chain of Xiph codec libraries (Ogg → Vorbis/FLAC/Opus). When `apt install` is unavailable (e.g., AI Studio), all must be built from source via FetchContent.

## Key Pitfalls

### 1. Cache variables must be set BEFORE FetchContent_MakeAvailable
libsndfile's `cmake/FindOgg.cmake`, `FindFLAC.cmake`, `FindVorbis.cmake`, `FindOpus.cmake` use `find_path`/`find_library`. If cache variables are set AFTER `FetchContent_MakeAvailable`, the find modules run during `MakeAvailable` and fail because the libraries haven't been built yet.

**Fix:** Use FetchContent naming convention (`_deps/<name>-src`, `_deps/<name>-build`) to pre-compute paths:

```cmake
set(_fc_base "${CMAKE_BINARY_DIR}/_deps")
set(OGG_INCLUDE_DIR "${_fc_base}/ogg-src/include" CACHE PATH "" FORCE)
set(OGG_LIBRARY "${_fc_base}/ogg-build/${CMAKE_STATIC_LIBRARY_PREFIX}ogg${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE FILEPATH "" FORCE)
# ... same for FLAC, Vorbis (vorbis, vorbisenc, vorbisfile), Opus
```

### 2. Imported targets must be created manually
When cache variables are pre-set, the find modules hit the "Already in cache, be silent" guard and skip creating imported targets (e.g., `FLAC::FLAC`). But libsndfile's `target_link_libraries` references them.

**Fix:** Create imported targets manually after setting cache vars:
```cmake
foreach(_spec "FLAC::FLAC|FLAC_LIBRARY|FLAC_INCLUDE_DIRS" ...)
    string(REPLACE "|" ";" _parts "${_spec}")
    list(GET _parts 0 _target)
    list(GET _parts 1 _lib_var)
    list(GET _parts 2 _inc_var)
    add_library(${_target} STATIC IMPORTED)
    set_target_properties(${_target} PROPERTIES
        IMPORTED_LOCATION "${${_lib_var}}"
        INTERFACE_INCLUDE_DIRECTORIES "${${_inc_var}}")
endforeach()
```

**Use `|` as separator, NOT `:`** — target names like `FLAC::FLAC` contain `::` which conflicts with `:`.

### 3. CMAKE_SKIP_INSTALL_RULES to avoid export conflicts
`FetchContent_MakeAvailable` triggers `install(EXPORT)` rules that fail when targets span multiple subdirectories (e.g., FLAC export depends on ogg not being in any export set).

**Fix:** Wrap the MakeAvailable call:
```cmake
set(_saved ${CMAKE_SKIP_INSTALL_RULES})
set(CMAKE_SKIP_INSTALL_RULES ON)
FetchContent_MakeAvailable(ogg vorbis FLAC opus)
set(CMAKE_SKIP_INSTALL_RULES ${_saved})
```

### 4. Ogg needs binary dir for generated config_types.h
`ogg/config_types.h` is generated during build, lives in `${ogg_BINARY_DIR}/include/`, not in the source tree.

**Fix:**
```cmake
set_target_properties(Ogg::ogg PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_fc_base}/ogg-src/include;${_fc_base}/ogg-build/include")
```

### 5. Opus header directory mismatch
Opus puts headers directly in `include/opus.h`, but code expects `include/opus/opus.h` (standard install layout).

**Fix:** Create a symbolic link:
```cmake
if(NOT EXISTS "${_fc_base}/opus-src/include/opus")
    file(CREATE_LINK "${_fc_base}/opus-src/include" "${_fc_base}/opus-src/include/opus" SYMBOLIC)
endif()
```

### 6. HAVE_EXTERNAL_XIPH_LIBS is unconditional
Line 127 of libsndfile's CMakeLists.txt: `set(HAVE_EXTERNAL_XIPH_LIBS ${ENABLE_EXTERNAL_LIBS})`. This means ALL xiph libs (including Opus) are linked, not just the ones found. You must provide ALL of them.

### 7. Proxy for AI Studio
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
