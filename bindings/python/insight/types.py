"""Core types: DType, Place, Shape, Slice.

These types form the foundation of the Insight7 type system.

DType: Data type enum for arrays.
    Available: bool, uint8, int8, int16, int32, int64,
    uint16, uint32, uint64, float16, bfloat16,
    float32, float64, complex64, complex128

Place: Device placement descriptor.
    CPUPlace()  — CPU device
    GPUPlace(id) — GPU device (default id=0)

Shape: Array shape descriptor.
    Shape([2, 3, 4]) — 3D shape with dims 2, 3, 4

Slice: Slice descriptor for array indexing.
    Slice(start, stop, step)
"""

from ._insight import DType, Place, Shape, Slice, CPUPlace, GPUPlace

__all__ = ["DType", "Place", "Shape", "Slice", "CPUPlace", "GPUPlace"]
