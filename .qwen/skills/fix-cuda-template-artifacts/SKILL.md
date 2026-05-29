---
name: fix-cuda-template-artifacts
description: Fix corrupted CUDA kernels caused by template rendering artifacts (double braces `{{}}` instead of single braces `{}`).
source: auto-skill
extracted_at: '2026-05-29T07:30:49.130Z'
---

# Fix CUDA Template Artifacts

When CUDA kernels are generated or modified by templating engines, they sometimes leave **double braces `{{}}`** instead of single braces `{}`. This causes `expected a declaration` or syntax errors during compilation (`nvcc`).

## Symptoms
- Build fails with `error: expected a declaration`.
- Errors are often at the start of blocks: `extern "C" {{`, `if (...) {{`, `for (...) {{`.
- Kernel registration variables are marked "declared but never referenced" because the code block above them failed to parse.

## Solution
1. **Identify affected files**: Use `grep` to find files with double braces.
   ```bash
   grep -rn "\{\{" backends/cuda/kernels/ --include="*.cu"
   ```
2. **Replace double braces**: Edit the files to replace `{{` with `{` and `}}` with `}`.
   - Be careful not to replace valid double braces if they are part of a string literal or specific macro syntax (rare in this codebase).
   - In this project, it is safe to globally replace `{{` -> `{` and `}}` -> `}` in the affected `.cu` files.

## Example
**Corrupted:**
```cpp
extern "C" {{
C_Status my_kernel_gpu(...) {{
    if (x) {{
        // logic
    }}
}}
```

**Fixed:**
```cpp
extern "C" {
C_Status my_kernel_gpu(...) {
    if (x) {
        // logic
    }
}
```
