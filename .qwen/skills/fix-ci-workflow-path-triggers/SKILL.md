---
name: fix-ci-workflow-path-triggers
description: Fix CI workflows not triggering on PRs when new directories are added (backends/**, tests/**, etc.)
source: auto-skill
extracted_at: '2026-06-02T11:08:13.174Z'
---

# Fix CI Workflow Path Triggers

## Problem

When adding new source directories (e.g., `backends/`, `bindings/`, `demos/`), CI
workflows with `paths:` filters in their `pull_request:` trigger may not fire on
PRs that modify files in the new directory.

The `push:` trigger often has more paths than `pull_request:`. When a PR only
touches `backends/cuda/kernels/signal/`, workflows missing `backends/**` in
their `pull_request.paths` won't run.

## Root Cause

GitHub Actions `paths:` filters use glob patterns. If a workflow has:
```yaml
on:
  push:
    paths: ['bindings/**', 'include/**', 'src/**', 'backends/**']
  pull_request:
    paths: ['bindings/**', 'include/**', 'src/**']  # ← missing backends/**
```

A PR modifying `backends/cuda/kernels/signal/windows/hann.cu` will trigger on
`push` (after merge) but NOT on `pull_request` (during review).

## Fix

Ensure `pull_request.paths` matches `push.paths` for all workflows:

```yaml
on:
  push:
    branches: [main]
    paths:
      - 'bindings/**'
      - 'include/**'
      - 'src/**'
      - 'backends/**'
      - 'CMakeLists.txt'
      - '.github/workflows/THIS_FILE.yml'
  pull_request:
    branches: [main]
    paths:
      - 'bindings/**'
      - 'include/**'
      - 'src/**'
      - 'backends/**'
      - 'CMakeLists.txt'
      - '.github/workflows/THIS_FILE.yml'
```

**Key**: Always include `.github/workflows/THIS_FILE.yml` in both triggers so
changes to the workflow itself are tested.

## Workflows to Check

When adding a new top-level directory, audit ALL workflow files:

```bash
grep -l "pull_request:" .github/workflows/*.yml
```

For each, compare `push.paths` vs `pull_request.paths` and add any missing entries.

## Common Missing Paths

| New Directory | Workflows to Update |
|---------------|-------------------|
| `backends/**` | cpu_tests, language_bindings, demo_tests |
| `tests/**` | cpu_tests, language_bindings |
| `demos/**` | demo_tests |
| `patches/**` | plot_tests |
| `cmake/**` | cpu_tests, plot_tests |

## Verification

After fixing, the workflow should show in the PR's "Checks" tab. If it doesn't
appear, the path filter is still wrong.
