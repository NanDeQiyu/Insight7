"""Basic operations demo for Insight7 Python bindings.

Demonstrates: array creation, arithmetic, multi-dtype, broadcasting,
unary operations, and reductions.
"""

import sys
import os
import math

# Add bindings to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "build", "bindings", "python")
)

import insight as ins
import numpy as np


def separator(title):
    print(f"\n{'=' * 40}")
    print(f"  {title}")
    print(f"{'=' * 40}")


def main():
    ins.init(["cpu"])

    print("Insight7 Basic Operations Demo (Python)")

    # --- Array creation ---
    separator("Array Creation")

    a = ins.ones(ins.Shape([3, 4]), ins.float32)
    print(f"ones(3,4) shape=({a.shape[0]},{a.shape[1]}) dtype={a.dtype}")

    b = ins.arange(12, dtype=ins.float64).reshape(ins.Shape([3, 4]))
    print(f"arange(0,12).reshape(3,4):\n{b.numpy()}")

    c = ins.eye(4, dtype=ins.float64)
    print(f"eye(4) F64:\n{c.numpy()}")

    # --- Multi-dtype arithmetic ---
    separator("Multi-Dtype Arithmetic")

    # F32
    f32_a = ins.from_array([1.0, 2.0, 3.0]).to(ins.float32)
    f32_b = ins.from_array([4.0, 5.0, 6.0]).to(ins.float32)
    f32_c = f32_a + f32_b
    print(f"F32: [1,2,3] + [4,5,6] = {f32_c.numpy()}")

    # F64
    f64_a = ins.from_array([1.0, 2.0, 3.0])
    f64_b = ins.from_array([0.5, 0.5, 0.5])
    f64_c = f64_a * f64_b
    print(f"F64: [1,2,3] * [0.5,0.5,0.5] = {f64_c.numpy()}")

    # I32
    i32_a = ins.from_array([10, 20, 30]).to(ins.int32)
    i32_b = ins.from_array([3, 3, 3]).to(ins.int32)
    i32_c = i32_a / i32_b
    print(f"I32: [10,20,30] / [3,3,3] = {i32_c.to(ins.int32).numpy()}")

    # --- Broadcasting ---
    separator("Broadcasting")

    m = ins.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    row = ins.from_array([[10.0, 20.0, 30.0]])
    result = m + row
    print(f"Matrix + Row broadcast:\n{result.numpy()}")

    # --- Unary operations ---
    separator("Unary Operations")

    x = ins.from_array([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"x:      {x.numpy()}")
    print(f"abs(x): {ins.abs(x).numpy()}")
    print(f"sin(x): {ins.sin(x).numpy()}")
    print(f"exp(x): {ins.exp(x).numpy()}")

    # --- Reductions ---
    separator("Reductions")

    data = ins.from_array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"data:  {data.numpy()}")
    print(f"sum:   {ins.sum(data).numpy()}")
    print(f"mean:  {ins.mean(data).numpy()}")
    print(f"max:   {ins.max(data).numpy()}")
    print(f"min:   {ins.min(data).numpy()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
