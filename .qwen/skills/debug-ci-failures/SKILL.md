---
name: debug-ci-failures
description: Systematic approach to debug and fix multi-job CI failures from user-provided log files
source: auto-skill
extracted_at: '2026-05-31T11:00:00.000Z'
---

# Debug CI Failures Systematically

When CI fails across multiple jobs, the user puts log files in `~/plum/texts/<project>/`.
Each file corresponds to a different CI job/workflow.

## Step 1: Read ALL log files

```bash
ls ~/plum/texts/<project>/
# Read every .txt file — don't skip any, even "old" ones (user may have updated them)
```

User explicitly expects all files to be read, not just new/changed ones.

## Step 2: Categorize failures by root cause

Group errors across jobs. Common patterns in Insight7:

| Symptom | Root Cause | Skill |
|---------|-----------|-------|
| `include could not find requested file: cmake/X.cmake` | `.gitignore` `*.cmake` blocks cmake/ modules | fix-ci-precommit |
| `Executable 'X' not found` in pre-commit | Local hook tool missing in CI | fix-ci-precommit |
| `F841 Local variable assigned but never used` | Unused variable (ruff lint) | fix-ci-precommit |
| `install(EXPORT) includes target X requires target Y` | Missing `CMAKE_SKIP_INSTALL_RULES=ON` for FetchContent | fetchcontent-xiph-codecs |
| `wrong number of template arguments` in third-party lib | Patch not applied (wrong path or not tracked) | apply-third-party-patch |
| `fatal error: X.h: No such file or directory` | Missing system package in CI workflow | fix-ci-precommit |
| `Patch file not found` | `patches/` directory not committed | fix-ci-precommit |
| `No rule to make target '_deps/X-build/lib/libY.a'` | IMPORTED target file path can't trigger build; need ALIAS target | fetchcontent-xiph-codecs |
| `terminate called after throwing ... not available` in Lua | `load_backend` binding doesn't catch C++ exceptions; add try-catch | fix-ci-precommit |
| IO tests pass locally, fail in CI | `write_bin` ofstream not explicitly closed; CI filesystem delays | fix-ci-precommit |
| `libinsight_cpu_backend.so: cannot open shared object` in Lua 5.3 | `--target insight_lua` doesn't build CPU backend; add `--target insight_cpu_backend` | fix-ci-precommit |
| `LfilterZi*` or `Mvdr*` fail in No-OpenBLAS build | Tests use LAPACK (solve/inv) internally; add `#ifndef INSIGHT_USE_OPENBLAS GTEST_SKIP()` | fix-ci-precommit |
| `No rule to make target '_deps/vorbis-build/lib/libY.a'` + build succeeds locally | `_resolve_dep` macro save/restore bug OR need ALIAS targets for codec deps | fix-ci-precommit |
| Lua GPU tests crash / abort in CI | `load_backend("cuda")` throws + GPU tests not guarded by `cuda_available` check | fix-ci-precommit |
| Different IO tests fail each CI run | Per-test SetUp/TearDown directory create/remove race; use suite-level fixture | fix-ci-precommit |
| `TearDownTestSuite: filesystem error: cannot remove: Directory not empty` | `std::filesystem::remove_all` throws when dir already removed by parallel run | fix-ci-precommit |
| `broadcast_to: cannot broadcast shape [N, 1] to [N]` in spectrogram/stft | `scatter` source must be 1D; `reshape(col, {freq_len, 1})` causes 2D→1D broadcast fail | debug-ci-failures |
| Julia: `No such file or directory: build/bindings/julia/modules/signal/windows.jl` | CMakeLists.txt only copies Insight.jl, not modules/ subdirectory | debug-ci-failures |
| Lua demo: `bad argument #2 to 'format' (number expected, got userdata)` | `string.format("%.3f", arr)` fails when arr is Insight Array userdata; use `ins.item(arr, 0)` to extract scalar | debug-ci-failures |
| Demo CI shows WARN but passes | `|| echo "WARN: ..."` masks failures; remove to let demos fail loudly | debug-ci-failures |
| `[insight] Failed to load library 'libinsight_cuda_backend.so'` in demos | Demos call `init({"cpu","cuda"})` which tries CUDA; change to `init({"cpu"})` for CPU-only CI | debug-ci-failures |
| Language binding CI not triggered on PR | `pull_request.paths` missing `backends/**`; must match `push.paths` | fix-ci-workflow-path-triggers |

## Step 3: Fix in dependency order

Some fixes depend on others:
1. First: gitignore / tracking issues (files need to exist in repo)
2. Second: CMake configuration issues (build must configure)
3. Third: Compilation issues (code must compile)
4. Fourth: Lint/format issues (style must pass)

## Step 4: Verify locally before committing

```bash
# Pre-commit
PATH="$HOME/.local/bin:$PATH" pre-commit run --all-files

# CMake configure (without building, to catch config errors fast)
rm -rf build && mkdir build && cd build
cmake .. -DINSIGHT_WITH_CUDA=OFF -DINSIGHT_BUILD_DEMOS=OFF -DINSIGHT_BUILD_TESTS=OFF 2>&1 | grep -E "Error|Warning"
```

## Step 5: Commit and let user push

```bash
git add -u
git commit -m "fix(ci): <description>"
# Do NOT git push — user handles that (needs password)
```

## Multi-round pattern

CI fixes often require multiple rounds because:
- Fixing one job unblocks the next set of errors
- The user updates the log files between rounds
- Some errors only appear after other errors are resolved

After each round: re-read ALL log files, don't assume previous content.
