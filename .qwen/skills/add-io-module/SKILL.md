---
name: add-io-module
description: How to add a new I/O module (like audio/sndfile) to Insight7 with namespace, tests, and demo
source: auto-skill
extracted_at: '2026-05-29T19:31:53.038Z'
---

# Adding a New I/O Module to Insight7

## File Structure

```
include/insight/io/<module>.h      # Public header
src/io/<module>.cpp                 # Implementation
tests/cpu/test_<module>.cpp         # CPU tests
tests/cuda/test_<module>.cpp        # CUDA tests (optional)
demos/<module>_demo.cpp             # Demo program
```

## 1. Header (`include/insight/io/<module>.h`)

Use a sub-namespace under `ins`:
```cpp
#pragma once
#include "insight/core/array.h"
namespace ins {
namespace audio {  // or whatever the module name is
    struct AudioInfo { ... };
    std::pair<Array, AudioInfo> read(const std::string &path);
    void write(const std::string &path, const Array &data, int samplerate = 44100);
} // namespace audio
} // namespace ins
```

## 2. Implementation (`src/io/<module>.cpp`)

Guard with the CMake define:
```cpp
#ifdef INSIGHT_USE_SNDFILE
#include "insight/io/sndfile.h"
#include "insight/core/exception.h"
#include "insight/ops/creation.h"
// ... implementation ...
#endif
```

Key patterns:
- Always convert to `CPUPlace()` before accessing raw data
- Use `to_array<T>(buffer, shape, CPUPlace())` for creating arrays from raw data
- Use `data.to(DType::F32).to(CPUPlace()).contiguous()` before writing

## 3. CPU Tests (`tests/cpu/test_<module>.cpp`)

Must call `ins::init()` in SetUpTestSuite:
```cpp
#include "insight/init.h"  // Required!

class AudioTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { ins::init(); }
};

TEST_F(AudioTest, ReadWriteMono) { ... }
```

**Common pitfall:** Forgetting `ins::init()` causes "Device interface not available" errors because `to_array` uses `get_device()` as default place.

## 4. CUDA Tests (`tests/cuda/test_<module>.cpp`)

Pattern: read on CPU → transfer to GPU → compute → transfer back → verify:
```cpp
class AudioTestGPU : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ins::init({"cpu", "cuda"});
        try { set_device(GPUPlace(0)); }
        catch (...) { GTEST_SKIP() << "GPU not available"; }
    }
};
```

**Avoid complex dtype operations on GPU:** `sum` doesn't support C32. Use `view(DType::F32)` or round-trip through FFT instead.

## 5. Demo (`demos/<module>_demo.cpp`)

Must call `ins::init({"cpu"})` at start. Use `#ifdef` guard and provide fallback `main()` that prints error.

Pass source directory via CMake:
```cmake
# demos/CMakeLists.txt
if(INSIGHT_USE_SNDFILE)
    add_insight_demo(demo_sndfile sndfile_demo.cpp)
    target_compile_definitions(demo_sndfile PRIVATE
        AUDIO_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../audio")
endif()
```

## 6. CMakeLists.txt Integration

Add option after FFTW3, before Include Directories:
```cmake
option(INSIGHT_USE_SNDFILE "Use libsndfile for audio file I/O" ON)
if(INSIGHT_USE_SNDFILE)
    # ... find or FetchContent ...
    target_link_libraries(insight_core PRIVATE SndFile::sndfile-static)
    target_compile_definitions(insight_core PUBLIC INSIGHT_USE_SNDFILE)
endif()
```

## 7. insight.h Integration

Add conditional include:
```cpp
#ifdef INSIGHT_USE_SNDFILE
#include "insight/io/sndfile.h"
#endif
```

## 8. Tests Auto-Discovery

CPU tests use `file(GLOB CPU_TEST_SOURCES "cpu/*.cpp")` — new test files are picked up automatically.

CUDA tests also use glob — no manual registration needed.

## 9. Running with Dynamic Backends

Backend DLLs must be in the same directory or `LD_LIBRARY_PATH`:
```bash
cd build/bin/demos && LD_LIBRARY_PATH="$(pwd):$LD_LIBRARY_PATH" ./demo_sndfile
```
