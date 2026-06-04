---
name: add-python-slice-getitem
description: Add py::slice overload to __getitem__ so Python native slice (x[:8]) works on Array
source: auto-skill
extracted_at: '2026-06-03T13:33:39.611Z'
---

# Python Native Slice Support for Array

## Problem

Python's `x[:8]` passes a native `slice` object to `__getitem__`, but pybind11 only
accepts the registered overloads (`str` or custom `Slice`). This causes:
```
TypeError: __getitem__(): incompatible function arguments.
The following argument types are supported:
    1. (self: Array, arg0: str) -> Array
    2. (self: Array, arg0: Slice) -> Array
```

## Solution

Add a `py::slice` overload to the `__getitem__` binding in `bindings/python/insight_py.cpp`:

```cpp
.def("__getitem__",
     [](const Array &a, py::slice pyslice) {
         py::ssize_t start, stop, step, slicelength;
         pyslice.compute(a.numel(), &start, &stop, &step, &slicelength);
         return a[Slice((int64_t)start, (int64_t)stop, (int64_t)step)];
     })
```

Place it after the existing `__getitem__` overloads (string and Slice).

## Why not `py::implicitly_convertible`?

`ins::Slice` uses `std::optional<int64_t>` members, which pybind11 can't auto-convert
from Python's `slice` (which uses `None` for missing values). The explicit lambda
approach is more reliable.

## How to apply

When adding `__getitem__` to any pybind11-bound class that should support Python slicing:
1. Add the `py::slice` overload alongside string/Slice overloads
2. Use `pyslice.compute(length, ...)` to get start/stop/step
3. Convert to your C++ Slice type
4. Test with: `arr[:8]`, `arr[::2]`, `arr[1:5:2]`
