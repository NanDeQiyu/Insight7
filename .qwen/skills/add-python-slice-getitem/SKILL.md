---
name: add-python-slice-getitem
description: Python native __getitem__ with int, tuple (mixed int/slice), and slice support for NumPy-style indexing
source: auto-skill
extracted_at: '2026-06-05T14:54:32.756Z'
---

# Python Native Indexing for Array (NumPy-style)

## Problem

Python's `a[1]`, `a[1, :]`, `a[:, 1]`, `a[1:3, 2]` all pass different types to
`__getitem__`: int, tuple (with mixed int/slice elements), and native slice.

## Solution: Four `__getitem__` overloads

In `bindings/python/insight_py.cpp`, register in this order:

### 1. Integer index — `a[1]`

```cpp
.def("__getitem__",
     [](const Array &a, int64_t idx) { return a[idx]; })
```

Calls C++ `operator[](int64_t)` which delegates to `at({idx})`.
For partial indexing (fewer indices than ndim), returns a slice view.

### 2. Tuple index — `a[1, :]`, `a[:, 1]`, `a[1:3, 2]`, `a[1, 3]`

```cpp
.def("__getitem__",
     [](const Array &a, py::tuple tup) {
       std::string spec;
       for (size_t i = 0; i < tup.size(); ++i) {
         if (i > 0) spec += ',';
         py::handle item = tup[i];
         if (py::isinstance<py::slice>(item)) {
           py::object s_start = item.attr("start");
           py::object s_stop = item.attr("stop");
           py::object s_step = item.attr("step");
           auto to_str = [](py::object o) -> std::string {
             if (o.is_none()) return "";
             return std::to_string(o.cast<int64_t>());
           };
           spec += to_str(s_start) + ':' + to_str(s_stop);
           if (!s_step.is_none()) spec += ':' + to_str(s_step);
         } else {
           spec += std::to_string(item.cast<int64_t>());
         }
       }
       return a[spec];
     })
```

**Key**: Build a comma-separated spec string from the tuple, then delegate to
`operator[](const std::string&)`. This reuses the existing multi-dimensional
string slicing logic. Use `item.attr("start")` etc. to preserve `None` values
(rather than `compute()` which requires a concrete length).

### 3. String spec — `a["1:3,2"]`

```cpp
.def("__getitem__",
     [](const Array &a, const std::string &spec) { return a[spec]; })
```

### 4. Native Python slice — `a[:8]` (single-dim)

```cpp
.def("__getitem__",
     [](const Array &a, py::slice pyslice) {
       py::object s_start = pyslice.attr("start");
       py::object s_stop = pyslice.attr("stop");
       py::object s_step = pyslice.attr("step");
       std::optional<int64_t> start =
           s_start.is_none() ? std::nullopt
                             : std::make_optional(s_start.cast<int64_t>());
       std::optional<int64_t> stop =
           s_stop.is_none() ? std::nullopt
                            : std::make_optional(s_stop.cast<int64_t>());
       int64_t step = s_step.is_none() ? 1 : s_step.cast<int64_t>();
       return a[Slice(start, stop, step)];
     })
```

**Important**: Use `.attr("start")`/`.attr("stop")`/`.attr("step")` to extract
`None` values as `std::nullopt`, NOT `pyslice.compute(length)` which forces
concrete values and loses the "open-ended" semantics.

## Why build string spec for tuples?

Converting tuple elements to a string spec (`"1,:"` → `a["1,:"]`) reuses the
existing `operator[](const std::string&)` which already handles:
- Partial indexing (fewer indices than ndim → remaining dims are full slices)
- Mixed int+slice dimensions
- Negative indices
- Squeeze of integer-indexed dimensions

## C++ prerequisite: partial `at()` support

The C++ `at(const std::vector<int64_t>&)` must support `indices.size() < ndim`:
- `indices.size() == ndim` → scalar view (original behavior)
- `indices.size() < ndim` → slice view, remaining dims kept as full slices

## How to apply

1. Ensure C++ `at()` supports partial indexing
2. Add all four `__getitem__` overloads in the order shown
3. Test with: `a[1]`, `a[-1]`, `a[1,3]`, `a[1,:]`, `a[:,1]`, `a[1:3,2]`,
   `a[:,::2]`, `a[::2,1:4]`, `a[0,-1]`, `a[:8]`, `a[::-1]`
