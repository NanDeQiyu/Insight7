---
name: fix-ci-doc-path-triggers
description: Exclude .md/.txt files from triggering build CI workflows — bindings/**/*.md should not trigger language_bindings or demo_tests
source: auto-skill
extracted_at: '2026-06-05T19:13:01.355Z'
---

# CI Path Exclusion for Documentation Files

## Problem

`bindings/**` path trigger in CI workflows catches README changes, triggering unnecessary builds.

## Solution

Add `!bindings/**/*.md` and `!bindings/**/*.txt` exclusions to workflow path triggers.

### language_bindings.yml

```yaml
on:
  push:
    branches: [ main, master, develop ]
    paths:
      - 'bindings/**'
      - '!bindings/**/*.md'
      - '!bindings/**/*.txt'
      - 'include/**'
      - 'src/**'
      - 'backends/**'
      # ...
  pull_request:
    branches: [ main, master ]
    paths:
      - 'bindings/**'
      - '!bindings/**/*.md'
      - '!bindings/**/*.txt'
      # ...
```

### demo_tests.yml

Same pattern — add exclusions to both push and pull_request triggers.

### Which workflows DON'T need this

- `cpu_tests.yml` — doesn't trigger on `bindings/**` at all
- `plot_tests.yml` — only triggers on plot-specific files
- `code_style.yml` — should run on ALL PRs (including docs)

## How to apply

1. Edit `.github/workflows/language_bindings.yml` — add `!bindings/**/*.md` and `!bindings/**/*.txt` to both push and pull_request paths
2. Edit `.github/workflows/demo_tests.yml` — same
3. Don't touch `code_style.yml` (it should run on all PRs)
