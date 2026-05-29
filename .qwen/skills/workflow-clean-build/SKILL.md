---
name: workflow-clean-build
description: Perform a clean build to resolve transient compilation issues.
source: auto-skill
extracted_at: '2026-05-29T08:00:00.000Z'
---

# Workflow: Clean Build

Sometimes `make` fails with confusing errors (e.g., type mismatches in unchanged files, linker errors) even when the source code seems correct. This is often due to stale object files or corrupted build state.

## Command
```bash
cd build && make clean && make -j24
```

## Why
- `make` only recompiles changed files. If a header changes or a dependency breaks, partial builds can fail.
- `make clean` removes all object files and forces a full recompilation.
- This is much faster than deleting the whole `build` directory and re-running `cmake`.

## Usage
If you encounter strange build errors, especially after editing headers or multiple files, perform a clean build before investigating code issues.
