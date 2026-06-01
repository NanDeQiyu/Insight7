"""Array class — core data container in Insight7.

The Array class supports:
    - Multi-dimensional indexing: arr["1:3, :"] or arr["0"]
    - Arithmetic operators: arr + arr, arr * 2.0, -arr
    - Comparison operators: arr == arr, arr < arr
    - Device transfer: arr.to(CPUPlace()), arr.to(GPUPlace(0))
    - NumPy interop: arr.numpy(), ins.from_numpy(np_arr)
    - Shape manipulation: arr.reshape([2,3]), arr.T, arr.squeeze()

Properties:
    arr.shape      — Shape object (tuple of dimension sizes)
    arr.dtype      — DType enum value
    arr.place      — Place object (CPUPlace or GPUPlace)
    arr.numel      — Total number of elements
    arr.nbytes     — Total bytes of data
    arr.ndim       — Number of dimensions
    arr.is_contiguous — Whether data is contiguous
    arr.defined    — Whether array is valid

Methods:
    arr.contiguous()   — Return contiguous copy if needed
    arr.reshape(shape) — Return view with new shape
    arr.transpose()    — Return transposed view
    arr.squeeze([axis])— Remove size-1 dimensions
    arr.unsqueeze(axis)— Insert size-1 dimension
    arr.to(place)      — Transfer to device
    arr.copy()         — Deep copy
    arr.numpy()        — Convert to NumPy array
"""

from ._insight import Array

__all__ = ["Array"]
