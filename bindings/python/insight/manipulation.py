"""Array manipulation operations.

Functions:
    concat, stack, vstack, hstack, split, tile, repeat
    flip, pad, reshape, flatten, ravel, squeeze, unsqueeze
    roll, permute, swapaxes, moveaxis
    fliplr, flipud, rot90, diag, diagonal, tril, triu, diff

Note: Array.transpose() is available as a method on the Array class.
Use arr.T for quick transpose, or ins.permute(arr, axes) for custom axes.
"""

from insight._insight import Shape
from insight._insight import (
    concat, stack, vstack, hstack, split, tile, repeat,
    flip, pad, flatten, ravel, squeeze, unsqueeze,
    roll, permute, swapaxes, moveaxis,
    fliplr, flipud, rot90, diag, diagonal, tril, triu, diff,
)
from insight._insight import reshape as _native_reshape


def reshape(x, new_shape):
    """Reshape an array to a new shape.

    Args:
        x: Input array.
        new_shape: Target shape as list/tuple of ints or Shape object.

    Returns:
        View with the new shape (shares data).
    """
    if isinstance(new_shape, (list, tuple)):
        new_shape = Shape(list(new_shape))
    return _native_reshape(x, new_shape)


__all__ = [
    "concat", "stack", "vstack", "hstack", "split", "tile", "repeat",
    "flip", "pad", "reshape", "flatten", "ravel", "squeeze", "unsqueeze",
    "roll", "permute", "swapaxes", "moveaxis",
    "fliplr", "flipud", "rot90", "diag", "diagonal", "tril", "triu", "diff",
]
