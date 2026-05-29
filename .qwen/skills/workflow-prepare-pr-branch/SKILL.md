---
name: workflow-prepare-pr-branch
description: Prepare clean PR branch(es) from accumulated work — supports single PR, split PRs, and squash-merge scenarios.
source: auto-skill
extracted_at: '2026-05-29T13:49:37.337Z'
---

## When to use

When accumulated work on a branch has grown beyond the original branch name's scope, or when you need to consolidate uncommitted changes into a clean PR-ready branch. Also covers splitting one branch into multiple PRs, or merging multiple commits/branches into a single PR.

## Files to always exclude

For this project, **never commit**:
- `.qwen/` — Qwen IDE config and skills (local tooling)
- `QWEN.md` — Qwen project config
- `claude`, `github` — symlinks to hidden dirs (local convenience, per git rules)
- `old_code/` — reference code, not part of the new architecture
- `test`, `test.cpp` — scratch/temporary files

## Scenario A: Single PR from uncommitted changes

### 1. Assess current state

```bash
git log --oneline -10
git branch -a
git status --short
git diff --name-only HEAD
git ls-files --others --exclude-standard
```

### 2. Create new branch from current HEAD

```bash
git checkout -b feat/<descriptive-name>
```

If the old branch name is too narrow (e.g., `feat/cuda-fft` but work covers all CUDA modules), create a broader-named branch.

### 3. Stage, verify, commit

```bash
git add backends/cuda/kernels/<module>/ src/ops/<module>.cpp tests/cuda/test_<module>.cpp backends/cuda/CMakeLists.txt
git diff --staged --stat   # verify
git commit -m "feat: add CUDA kernels for <module list>"
```

### 4. Verify and inform user

```bash
git log --oneline -5
git status   # should show only intentionally excluded untracked files
```

Never run `git push`. Tell the user: branch name, target (`main`), PR summary (files, lines, test status).

## Scenario B: Split one branch into multiple PRs

When a branch has N commits that should become separate PRs (e.g., FFT in one PR, backend in another):

### 1. Create per-feature branches from main

```bash
# Branch for first PR (commit 1 only)
git branch feat/<first> <commit-1-hash>

# Branch for second PR (commits 1+2, will rebase later)
git branch feat/<second> <commit-2-hash>
```

### 2. Rebase second branch to exclude first commit

```bash
git rebase --onto main feat/<first> feat/<second>
```

This replays only commits after `feat/<first>` onto `main`.

### 3. Resolve conflicts

If `backends/cuda/CMakeLists.txt` conflicts (common when both branches modify `target_link_libraries`):
- **Split scenario**: keep only the dependencies the second PR actually needs (e.g., `CUDA::curand` but NOT `CUDA::cufft` which belongs to the first PR)
- Use `GIT_EDITOR=true git rebase --continue` (no `--no-edit` flag)

### 4. Verify isolation

```bash
git log --oneline feat/<first> --not main   # should show only commit 1
git log --oneline feat/<second> --not main  # should show only commit 2
```

## Scenario C: Squash multiple commits/branches into one PR

When you have commits across branches that should become a single PR (e.g., user changes mind from "two PRs" to "one PR"):

### 1. Reset to main, cherry-pick in order

```bash
git checkout feat/<target-branch>
git reset --hard main
git cherry-pick <commit-1-hash>
git cherry-pick <commit-2-hash>
```

### 2. Resolve CMakeLists.txt conflicts during cherry-pick

If `backends/cuda/CMakeLists.txt` conflicts:
- **Merge scenario**: keep ALL dependencies from both commits (e.g., `CUDA::cufft CUDA::curand`)
- `git add backends/cuda/CMakeLists.txt && GIT_EDITOR=true git cherry-pick --continue`

### 3. Squash into one commit

```bash
git reset --soft main
git commit -m "feat: add CUDA backend kernels for FFT, elementwise, ..."
```

### 4. Verify

```bash
git log --oneline feat/<target-branch> --not main  # should show exactly 1 commit
git diff --stat main..feat/<target-branch>          # should include all files
```

## CMakeLists.txt dependency rules

When modifying `target_link_libraries` in `backends/cuda/CMakeLists.txt`:
- `CUDA::cufft` — needed by FFT kernels
- `CUDA::curand` — needed by random kernels
- `CUDA::cudart` — always needed
- `insight_core` — always needed

When splitting/merging PRs, ensure each PR has exactly the dependencies it needs (split) or all dependencies (merge).
