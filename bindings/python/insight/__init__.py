"""
Insight7 — Lightweight scientific computing framework for Python.

A NumPy-compatible tensor library with GPU acceleration support,
inspired by PaddlePaddle, Torch7, and NumPy/CuPy.

Quick Start::

    import insight as ins

    # Create arrays (Paddle-style API)
    a = ins.zeros([2, 3], ins.float32)
    b = ins.ones([2, 3], ins.float32)
    c = a + b

    # Reduction
    s = ins.sum(c, axis=0)

    # Linear algebra
    m = ins.matmul(a, b.T)

    # NumPy interop
    import numpy as np
    arr = a.numpy()              # Insight -> NumPy
    d = ins.from_numpy(arr)      # NumPy -> Insight

DType Shortcuts (Paddle-style)::

    ins.float32, ins.float64, ins.int32, ins.int64, ins.bool, ...

Place Constructors::

    ins.CPUPlace()       # CPU device
    ins.GPUPlace(0)      # GPU device 0
"""

try:
    from ._insight import *  # noqa: F401,F403
    from ._insight import (
        DType, Place, Shape, Slice, Array,
        init, is_initialized,
        # Creation
        zeros, ones, full, eye, arange, linspace, logspace,
        from_numpy, zeros_like, ones_like,
        # Elementwise binary
        add, sub, mul, div, pow, mod,
        maximum, minimum,
        equal, not_equal, greater, less, greater_equal, less_equal,
        logical_and, logical_or, logical_xor, logical_not,
        # Unary math
        abs, negative, square, sqrt, exp, log, log2, log10,
        sin, cos, tan, asin, acos, atan,
        sinh, cosh, tanh,
        floor, ceil, round, sign,
        isnan, isinf, isfinite, where,
        # Reduction
        sum, mean, max, min, prod, argmax, argmin,
        any, all, var, std, cumsum, cumprod,
        # Manipulation
        concat, stack, vstack, hstack, split, tile, repeat,
        flip, pad, reshape, flatten, ravel, squeeze, unsqueeze, roll,
        # Linalg
        matmul, dot, inner, outer,
        det, slogdet, inv, pinv, solve, lstsq,
        svd, svdvals, eigh, eigvalsh, eig, eigvals,
        cholesky, qr, lq, lu,
        norm, cond, matrix_rank, matrix_power, trace, cov,
        # FFT
        fft, ifft, fft2, ifft2, fftn, ifftn,
        rfft, irfft, fftfreq, rfftfreq,
        # Signal
        convolve, unwrap, sinc,
        # Random
        rand, randn, randint, normal, uniform, randperm,
        # Cast & Indexing
        cast, take, nonzero, flatnonzero, argsort, sort,
        masked_select, searchsorted,
        # Place constructors
        CPUPlace, GPUPlace,
    )

    # ── Dtype docstrings ──────────────────────────────────────────────
    # These are module-level attributes set by the native module:
    #   float32, float64, float16, bfloat16,
    #   int8, int16, int32, int64,
    #   uint8, uint16, uint32, uint64,
    #   bool, complex64, complex128

    # ── Function docstrings (for IDE autocompletion) ──────────────────

    def zeros(shape, dtype=float32, place=None):
        """Create an array filled with zeros.

        Args:
            shape: Array dimensions as a list/tuple, e.g. [2, 3].
            dtype: Data type (default: float32).
            place: Device placement (default: CPU).

        Returns:
            Array of zeros with the given shape and dtype.

        Example::

            a = ins.zeros([2, 3], ins.float32)
        """
        if place is None:
            place = CPUPlace()
        return _zeros(shape, dtype, place)

    def ones(shape, dtype=float32, place=None):
        """Create an array filled with ones.

        Args:
            shape: Array dimensions as a list/tuple.
            dtype: Data type (default: float32).
            place: Device placement (default: CPU).

        Returns:
            Array of ones with the given shape and dtype.
        """
        if place is None:
            place = CPUPlace()
        return _ones(shape, dtype, place)

    # Save original functions before overriding
    _zeros = zeros
    _ones = ones

except ImportError as e:
    raise ImportError(
        f"Failed to load Insight native module: {e}\n"
        "Make sure you have built the Python binding:\n"
        "  cmake --build build --target insight_python"
    ) from e

__version__ = "1.0.0"
__all__ = [
    # Types
    "DType", "Place", "Shape", "Slice", "Array",
    # Init
    "init", "is_initialized",
    # Place
    "CPUPlace", "GPUPlace",
    # Dtypes
    "float32", "float64", "float16", "bfloat16",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "bool", "complex64", "complex128",
    # Creation
    "zeros", "ones", "full", "eye", "arange", "linspace",
    "from_numpy", "zeros_like", "ones_like",
    # Elementwise
    "add", "sub", "mul", "div", "pow", "mod",
    "maximum", "minimum", "equal", "not_equal",
    "greater", "less", "greater_equal", "less_equal",
    "logical_and", "logical_or", "logical_xor", "logical_not",
    # Unary
    "abs", "negative", "square", "sqrt", "exp", "log",
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "floor", "ceil", "round", "sign",
    "isnan", "isinf", "isfinite", "where",
    # Reduction
    "sum", "mean", "max", "min", "prod",
    "argmax", "argmin", "any", "all", "var", "std",
    # Manipulation
    "concat", "stack", "split", "tile", "repeat",
    "flip", "pad", "reshape", "flatten", "ravel",
    "squeeze", "unsqueeze", "roll",
    # Linalg
    "matmul", "dot", "outer", "det", "inv", "solve",
    "svd", "eigh", "cholesky", "qr", "norm", "trace",
    # FFT
    "fft", "ifft", "rfft", "irfft", "fftfreq",
    # Signal
    "convolve", "unwrap", "sinc",
    # Random
    "rand", "randn", "randint", "normal", "uniform",
    # Cast & Indexing
    "cast", "take", "nonzero", "argsort", "sort",
]
