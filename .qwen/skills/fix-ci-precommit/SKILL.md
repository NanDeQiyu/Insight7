---
name: fix-ci-precommit
description: Fix CI failures in code_style workflow — gitignore cmake modules, pre-commit local hooks, ruff lint errors, patches/ not tracked, libsndfile export set error
source: auto-skill
extracted_at: '2026-05-31T07:00:00.000Z'
---

# Fix CI Pre-commit / Code Style Failures

The `code_style.yml` workflow runs `pre-commit run --all-files` on every PR.
Three common failure patterns and their fixes:

## 1. `.gitignore` blocks cmake modules

**Symptom**: `CMake Error: include could not find requested file: cmake/ApplyPatch.cmake`

**Root cause**: `.gitignore` has `*.cmake` which matches files in any directory, silently preventing `cmake/*.cmake` modules from being committed. Other cmake files (FindFFTW3.cmake etc.) were force-added before the rule existed.

**Fix**:
```
# In .gitignore, after *.cmake:
!cmake/*.cmake
```

**Verify**:
```bash
git check-ignore cmake/ApplyPatch.cmake  # should exit 1 (not ignored)
git ls-files cmake/                       # should show ApplyPatch.cmake
```

## 2. Pre-commit local hook: tool not found in CI

**Symptom**: `Executable '/home/aistudio/.local/bin/stylua' not found`

**Root cause**: `.pre-commit-config.yaml` uses a `repo: local` hook with `language: system` and a hardcoded absolute path. Works locally but CI runner doesn't have the tool.

**Fix pattern** — two changes:

### a) `.pre-commit-config.yaml` — use bare command name
```yaml
- repo: local
  hooks:
    - id: stylua
      name: stylua
      entry: stylua --indent-type Spaces --indent-width 2  # NOT /home/.../stylua
      language: system
      files: \.lua$
      exclude: third_party/|build/|out/
```

### b) `.github/workflows/code_style.yml` — install the binary
```yaml
- name: Install stylua
  run: |
    curl -sL https://github.com/JohnnyMorganz/StyLua/releases/download/v2.1.0/stylua-linux-x86_64.zip -o stylua.zip
    unzip stylua.zip
    sudo mv stylua /usr/local/bin/
    rm stylua.zip
```

**Why not use a remote repo hook?** StyLua's official pre-commit repo uses `language: rust` which builds from source (slow, needs Rust toolchain). A pre-built binary download is faster and more reliable.

## 3. Ruff F841: unused variable

**Symptom**: `demos/radar_task1.py:32:9: F841 Local variable 'place' is assigned to but never used`

**Fix**: Remove the unused variable or use it. For `place = ins.CPUPlace()` / `ins.GPUPlace(0)` that's never referenced later, just delete both branches of the assignment.

## 4. `patches/` directory not tracked in git

**Symptom (plot-tests CI)**: `CMake Error at cmake/ApplyPatch.cmake:10: Patch file not found: .../patches/matplotplusplus/gcc-compat.patch`

**Root cause**: The `patches/` directory was never added to git. Unlike the `cmake/` issue (blocked by `.gitignore`), `patches/` has no gitignore rule — it was simply forgotten.

**Fix**:
```bash
git add patches/
```

**Verify**:
```bash
git ls-files patches/   # should show the patch file(s)
```

## 5. libsndfile `install(EXPORT)` error with FetchContent

**Symptom (CPU tests + Python binding CI)**:
```
CMake Error: install(EXPORT "targets" ...) includes target "FLAC" which requires target "ogg" that is not in any export set.
```

**Root cause**: `_resolve_dep` macro in CMakeLists.txt sets `CMAKE_SKIP_INSTALL_RULES=ON` for the `add_subdirectory` path but NOT for the `FetchContent_MakeAvailable` path. CI uses FetchContent (no local third_party/), so install rules are created for ogg/FLAC/vorbis/opus, and libsndfile's `install(EXPORT)` fails because ogg is missing from the export set.

**Fix**: Set `CMAKE_SKIP_INSTALL_RULES=ON` before every `FetchContent_MakeAvailable` for codec deps:
```cmake
# In the _resolve_dep macro, FetchContent path:
set(CMAKE_SKIP_INSTALL_RULES ON)
FetchContent_MakeAvailable(${NAME})
set(CMAKE_SKIP_INSTALL_RULES ${_saved_skip})
```

Also apply to the sndfile `FetchContent_MakeAvailable`:
```cmake
set(CMAKE_SKIP_INSTALL_RULES ON)
FetchContent_MakeAvailable(sndfile)
set(CMAKE_SKIP_INSTALL_RULES OFF)
```

**Why `ON` and not saved/restored?** For sndfile specifically, we don't want ANY install rules from subdependencies. Setting to `OFF` (default) after is fine since the parent project's install rules are set up later.

**Verify**: cmake configure should show no export errors, even without local third_party/ sources.

## 6. `apply_patch` targets wrong path with FetchContent

**Symptom (plot-tests, CPU tests, Python binding CI)**: matplotplusplus template compilation errors (`wrong number of template arguments (2, should be 1)` in `to_vector_2d<T1, unsigned char>`) even though `gcc-compat.patch` is committed.

**Root cause**: `apply_patch("${LOCAL_MATPLOT}", ...)` is called after the `if(local)/else(FetchContent)` block, using `LOCAL_MATPLOT` which always points to `third_party/matplotplusplus/`. When FetchContent downloads the source to `_deps/matplotplusplus-src/`, the patch targets a non-existent directory and silently fails (ApplyPatch.cmake outputs a WARNING but doesn't error).

**Fix**: Move `apply_patch` into each branch, using the correct source dir:
```cmake
if(EXISTS "${LOCAL_MATPLOT}/CMakeLists.txt")
    add_subdirectory("${LOCAL_MATPLOT}" ...)
    apply_patch("${LOCAL_MATPLOT}"              # ← local path
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/matplotplusplus/gcc-compat.patch")
else()
    FetchContent_MakeAvailable(matplotplusplus)
    apply_patch("${matplotplusplus_SOURCE_DIR}"  # ← FetchContent path
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/matplotplusplus/gcc-compat.patch")
endif()
```

**Verify**: cmake configure should show `Patch gcc-compat applied successfully` (not "check failed" or "not found").

## 7. CI workflow missing system dependencies

**Symptom (plot-tests CI)**: `fatal error: lapacke.h: No such file or directory`

**Root cause**: `plot_tests.yml` installs `libopenblas-dev` but not `liblapacke-dev`. The linalg kernels `#include <lapacke.h>`.

**Fix**: Add `liblapacke-dev` to the apt-get install step in the workflow.

**General pattern**: When adding new C++ features that depend on system libraries, check ALL CI workflows (not just cpu_tests.yml) for the required packages.

## 8. `_resolve_dep` macro save/restore bug

**Symptom**: `No rule to make target '_deps/vorbis-build/lib/libvorbisenc.a'`

**Root cause**: The `_resolve_dep` macro saves `_saved_skip` only in the `if(local)` branch. The `else(FetchContent)` branch uses the stale `_saved_skip` from a previous call, corrupting `CMAKE_SKIP_INSTALL_RULES` for subsequent dependencies.

**Fix**: Save `_saved_skip` at the top of the macro, before the if/else:
```cmake
macro(_resolve_dep VAR NAME)
    set(_saved_skip ${CMAKE_SKIP_INSTALL_RULES})   # ← save ONCE, before if/else
    if(DEFINED ${VAR})
        ...
    else()
        ...
    endif()
endmacro()
```

## 9. Codec deps need ALIAS targets, not IMPORTED

**Symptom**: `No rule to make target '_deps/vorbis-build/lib/libvorbisenc.a'` (same symptom as #8, different root cause)

**Root cause**: When codec deps are built via `add_subdirectory`, the real targets (`vorbisenc`, `FLAC`) exist in the global namespace, but libsndfile references them as `Vorbis::vorbisenc`, `FLAC::FLAC`. Our code creates IMPORTED targets with `IMPORTED_LOCATION` pointing to file paths. CMake can't resolve file path dependencies to build rules.

**Fix**: Create ALIAS targets when the base target exists:
```cmake
if(NOT TARGET ${_target})
    string(REPLACE "::" ";" _name_parts "${_target}")
    list(GET _name_parts 1 _base_name)
    if(TARGET ${_base_name})
        add_library(${_target} ALIAS ${_base_name})    # ← ALIAS, not IMPORTED
    else()
        add_library(${_target} STATIC IMPORTED)        # ← fallback for FetchContent
        ...
    endif()
endif()
```

Note: `ogg` and `opus` already create `Ogg::ogg` / `Opus::opus` aliases. `vorbis` and `FLAC` do NOT — they need manual aliases.

## 10. `load_backend` binding must catch exceptions

**Symptom (Lua CI)**: `terminate called after throwing ... insight_cuda_backendnot available` → core dump

**Root cause**: `ins::load_backend("cuda")` throws when `.so` not found. Lua binding doesn't catch, sol2 can't handle it, process aborts.

**Fix**: Lua binding returns bool, Python binding converts to `ValueError`:
```cpp
// Lua
m["load_backend"] = [](const std::string &backend) -> bool {
    try { ins::load_backend(backend); return true; }
    catch (...) { return false; }
};
// Python
try { ins::load_backend(backend); }
catch (const std::exception &e) { throw py::value_error(e.what()); }
```

## 11. GPU tests need CUDA availability guard

**Symptom (Lua CI)**: GPU tests crash when CUDA not available

**Fix pattern for Lua (busted)**:
```lua
local cuda_available = ins.load_backend("cuda")
it("array to GPU", function()
    if not cuda_available then pending("CUDA not available") return end
    -- ... GPU test code ...
end)
```

Device info tests should use `pcall`:
```lua
it("cuda_version returns number", function()
    local ok, v = pcall(ins.cuda_version)
    if ok then assert.is_number(v) end
end)
```

## 12. LAPACK-dependent tests need OpenBLAS guard

**Symptom (No-OpenBLAS CI)**: `LfilterZiFIR`, `LfilterZiIIR`, `MvdrBasic` fail

**Root cause**: These tests call `linalg::solve` or `inv` internally, which require LAPACK (from OpenBLAS). The tests have FFTW guards but not OpenBLAS guards.

**Fix**:
```cpp
TEST_F(Suite, TestName) {
#ifndef INSIGHT_USE_OPENBLAS
  GTEST_SKIP() << "OpenBLAS not available, skipping LAPACK-dependent test";
#endif
  // ... test code ...
}
```

## 13. IO test fixtures: suite-level setup, explicit file close

**Symptom (CI-only)**: IO tests pass locally, fail in CI. Different tests fail each run (505/506 one run, 507/508 next).

**Root causes**:
1. Per-test `SetUp`/`TearDown` creates/removes `/tmp/insight_io_test` — CI filesystem delay causes race
2. `write_bin` ofstream not explicitly closed — CI filesystem may not flush on destructor

**Fix**:
```cpp
// Suite-level fixture (create once, cleanup once)
class SignalIOTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu"});
    std::filesystem::create_directories("/tmp/insight_io_test");
  }
  static void TearDownTestSuite() {
    std::filesystem::remove_all("/tmp/insight_io_test");
  }
  void SetUp() override { tmp_dir = "/tmp/insight_io_test"; }
};

// write_bin: explicit close
ofs.write(...);
ofs.close();  // ← don't rely on destructor
```

## 14. Lua binding `--target` must include backend

**Symptom (Lua 5.3 CI)**: `libinsight_cpu_backend.so: cannot open shared object file`

**Root cause**: `cmake --build . --target insight_lua` only builds the Lua binding .so, not the CPU backend .so. The CPU backend is loaded dynamically at runtime by `ins::init()`.

**Fix**: Build both targets:
```yaml
run: cd build && cmake --build . -j$(nproc) --target insight_lua --target insight_cpu_backend
```

## Verification

After fixing, run locally:
```bash
PATH="$HOME/.local/bin:$PATH" pre-commit run --all-files
```

All 4 hooks should pass: `clang-format`, `ruff`, `ruff-format`, `stylua`.
