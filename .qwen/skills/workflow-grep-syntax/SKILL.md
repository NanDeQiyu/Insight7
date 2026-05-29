---
name: workflow-grep-syntax
description: Use grep to proactively find syntax anomalies (like double braces) in CUDA kernels before compiling.
source: auto-skill
extracted_at: '2026-05-29T07:30:49.130Z'
---

# Workflow: Grep for Syntax Anomalies

Before attempting to build after a batch edit or generation, it is highly efficient to run a quick `grep` check for common syntax anomalies that the compiler might complain about.

## Command
```bash
grep -rn "\{\{" backends/cuda/ --include="*.cu"
```

## Why
- Compiling CUDA kernels takes time.
- Template engines often produce double braces `{{` which are invalid C++.
- Catching this before `make` saves minutes of build-fail-fix cycles.

## Usage
After modifying or adding multiple `.cu` files, run the command. If it returns matches, fix them (see `fix-cuda-template-artifacts` skill) before running `make`.
