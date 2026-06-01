"""Type casting functions.

Provides functions for converting arrays between different data types.
"""

from insight._insight import cast as _cast

__all__ = [
    "cast",
]


def cast(x, dtype, copy=True):
    """Cast an array to a different data type.

    Args:
        x: Input array.
        dtype: Target data type (e.g., ``ins.float32``, ``ins.int64``).
        copy: If True (default), always return a new array.  If False,
            return the input array unchanged when the dtype already matches.

    Returns:
        Array with the specified data type.
    """
    return _cast(x, dtype, copy=copy)
