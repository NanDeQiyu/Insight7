---
name: fix-ci-binding-test-paths
description: Fix CI workflow test paths after tests/bindings/ migration to per-module structure
source: auto-skill
extracted_at: '2026-06-02T12:01:00.000Z'
---

# Fix CI Binding Test Paths

After migrating from `tests/bindings/` to per-module test structure (`tests/cpu/{python,lua,julia}/`), CI workflows need updating.

## Problem

CI workflows reference old test paths like `tests/bindings/test_python_binding.py` which no longer exist.

## Fix

### Python (pytest auto-discovers all test_*.py files)
```yaml
- name: Test Python binding
  run: |
    export PYTHONPATH=bindings/python
    export LD_LIBRARY_PATH=build/backends/cpu:$LD_LIBRARY_PATH
    python3 -m pytest tests/cpu/python/ -v --tb=short --ignore=tests/cpu/python/test_plot.py
```

Note: `--ignore=test_plot.py` because plot tests segfault in headless CI (matplotplusplus requires display).

### Lua (busted can discover all .lua files in a directory)
```yaml
- name: Test Lua binding
  run: |
    export LUA_CPATH="$GITHUB_WORKSPACE/build/bindings/lua/?.so;;"
    export LUA_PATH="$GITHUB_WORKSPACE/bindings/lua/?/init.lua;$GITHUB_WORKSPACE/bindings/lua/?.lua;;"
    export LD_LIBRARY_PATH=$GITHUB_WORKSPACE/build/backends/cpu:$LD_LIBRARY_PATH
    cd tests/cpu/lua && busted .
```

**CRITICAL**: Use `$GITHUB_WORKSPACE` absolute paths. Relative paths break when `cd tests/cpu/lua` changes the working directory.

### Julia (run all test_*.jl files)
```yaml
- name: Test Julia binding
  run: |
    cp bindings/julia/Insight.jl build/bindings/julia/Insight.jl
    cp -r bindings/julia/modules build/bindings/julia/modules
    export LD_LIBRARY_PATH=$GITHUB_WORKSPACE/build/backends/cpu:$GITHUB_WORKSPACE/build/bindings/julia:$LD_LIBRARY_PATH
    for f in tests/cpu/julia/test_*.jl; do julia "$f" || exit 1; done
```

**CRITICAL**: `LD_LIBRARY_PATH` must include `build/bindings/julia/` where `libinsight_julia.so` lives.

## Path trigger fix

Also update `pull_request.paths` to include `backends/**`:
```yaml
pull_request:
  paths:
    - 'bindings/**'
    - 'include/**'
    - 'src/**'
    - 'backends/**'
    - 'CMakeLists.txt'
    - '.github/workflows/<name>.yml'
```

## How to apply

After any test directory restructuring:
1. Update all CI workflow files that reference test paths
2. Use absolute paths (`$GITHUB_WORKSPACE`) when `cd` changes working directory
3. Include `build/bindings/julia/` in Julia's `LD_LIBRARY_PATH`
4. Run `pre-commit run --all-files` before committing
