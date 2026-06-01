"""Random number generation functions.

Provides functions for creating arrays filled with random values from
various distributions, as well as seeding and querying the random state.

All distribution functions accept a *shape*, *dtype*, and *place* to
control the output array properties.
"""

from insight._insight import (
    beta as _beta,
    binomial as _binomial,
    exponential as _exponential,
    gamma as _gamma,
    get_seed as _get_seed,
    normal as _normal,
    poisson as _poisson,
    rand as _rand,
    randint as _randint,
    rand_like as _rand_like,
    randn as _randn,
    randn_like as _randn_like,
    randperm as _randperm,
    seed as _seed,
    uniform as _uniform,
)

__all__ = [
    "rand",
    "randn",
    "randint",
    "normal",
    "uniform",
    "randperm",
    "seed",
    "get_seed",
    "rand_like",
    "randn_like",
    "exponential",
    "gamma",
    "beta",
    "binomial",
    "poisson",
]


def rand(shape, dtype=None, place=None):
    """Create an array with values uniformly distributed in [0, 1).

    Args:
        shape: Shape of the output array.
        dtype: Data type of the output.  Default is ``float32``.
        place: Device placement.  Default is the current device.

    Returns:
        Array of random values in [0, 1).
    """
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if place is not None:
        kwargs["place"] = place
    return _rand(shape, **kwargs)


def randn(shape, dtype=None, place=None):
    """Create an array with values from the standard normal distribution.

    Args:
        shape: Shape of the output array.
        dtype: Data type of the output.  Default is ``float32``.
        place: Device placement.  Default is the current device.

    Returns:
        Array of standard normal random values.
    """
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if place is not None:
        kwargs["place"] = place
    return _randn(shape, **kwargs)


def randint(low, high, shape, dtype=None, place=None):
    """Create an array with random integer values in [low, high).

    Args:
        low: Lowest integer to be drawn (inclusive).
        high: One above the highest integer to be drawn (exclusive).
        shape: Shape of the output array.
        dtype: Data type of the output.  Default is ``int64``.
        place: Device placement.  Default is the current device.

    Returns:
        Array of random integers.
    """
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if place is not None:
        kwargs["place"] = place
    return _randint(low, high, shape, **kwargs)


def normal(mean=0.0, std=1.0, shape=None, dtype=None, place=None):
    """Create an array with values from a normal distribution.

    Args:
        mean: Mean of the distribution.  Default is 0.0.
        std: Standard deviation.  Default is 1.0.
        shape: Shape of the output array.
        dtype: Data type of the output.  Default is ``float32``.
        place: Device placement.  Default is the current device.

    Returns:
        Array of normal random values.
    """
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if place is not None:
        kwargs["place"] = place
    return _normal(mean, std, shape, **kwargs)


def uniform(low=0.0, high=1.0, shape=None, dtype=None, place=None):
    """Create an array with values from a uniform distribution.

    Args:
        low: Lower bound of the distribution.  Default is 0.0.
        high: Upper bound of the distribution.  Default is 1.0.
        shape: Shape of the output array.
        dtype: Data type of the output.  Default is ``float32``.
        place: Device placement.  Default is the current device.

    Returns:
        Array of uniformly distributed random values.
    """
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if place is not None:
        kwargs["place"] = place
    return _uniform(low, high, shape, **kwargs)


def randperm(n, dtype=None, place=None):
    """Create a random permutation of integers from 0 to n-1.

    Args:
        n: Upper bound (exclusive).
        dtype: Data type of the output.  Default is ``int64``.
        place: Device placement.  Default is the current device.

    Returns:
        1-D array of permuted integers.
    """
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if place is not None:
        kwargs["place"] = place
    return _randperm(n, **kwargs)


def seed(s):
    """Set the random seed for reproducibility.

    Args:
        s: Integer seed value.
    """
    _seed(s)


def get_seed():
    """Get the current random seed.

    Returns:
        The current seed as an integer.
    """
    return _get_seed()


def rand_like(x):
    """Create a uniform random array with the same shape and dtype.

    Args:
        x: Template array whose shape and dtype are copied.

    Returns:
        Array of random values in [0, 1) with the same shape and dtype.
    """
    return _rand_like(x)


def randn_like(x):
    """Create a standard normal random array with the same shape and dtype.

    Args:
        x: Template array whose shape and dtype are copied.

    Returns:
        Array of standard normal values with the same shape and dtype.
    """
    return _randn_like(x)


def exponential(scale, shape, dtype=None, place=None):
    """Create an array with values from an exponential distribution.

    Args:
        scale: Scale parameter (1 / rate).
        shape: Shape of the output array.
        dtype: Data type of the output.  Default is ``float32``.
        place: Device placement.  Default is the current device.

    Returns:
        Array of exponentially distributed random values.
    """
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if place is not None:
        kwargs["place"] = place
    return _exponential(scale, shape, **kwargs)


def gamma(shape_p, rate, out_shape, dtype=None, place=None):
    """Create an array with values from a gamma distribution.

    Args:
        shape_p: Shape parameter (alpha) of the gamma distribution.
        rate: Rate parameter (1 / theta) of the gamma distribution.
        out_shape: Shape of the output array.
        dtype: Data type of the output.  Default is ``float32``.
        place: Device placement.  Default is the current device.

    Returns:
        Array of gamma-distributed random values.
    """
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if place is not None:
        kwargs["place"] = place
    return _gamma(shape_p, rate, out_shape, **kwargs)


def beta(a, b, shape, dtype=None, place=None):
    """Create an array with values from a beta distribution.

    Args:
        a: Alpha parameter of the beta distribution.
        b: Beta parameter of the beta distribution.
        shape: Shape of the output array.
        dtype: Data type of the output.  Default is ``float32``.
        place: Device placement.  Default is the current device.

    Returns:
        Array of beta-distributed random values.
    """
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if place is not None:
        kwargs["place"] = place
    return _beta(a, b, shape, **kwargs)


def binomial(n, p, shape, dtype=None, place=None):
    """Create an array with values from a binomial distribution.

    Args:
        n: Number of trials.
        p: Probability of success for each trial.
        shape: Shape of the output array.
        dtype: Data type of the output.  Default is ``int64``.
        place: Device placement.  Default is the current device.

    Returns:
        Array of binomially distributed random values.
    """
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if place is not None:
        kwargs["place"] = place
    return _binomial(n, p, shape, **kwargs)


def poisson(lam, shape, dtype=None, place=None):
    """Create an array with values from a Poisson distribution.

    Args:
        lam: Expected number of events per interval (lambda).
        shape: Shape of the output array.
        dtype: Data type of the output.  Default is ``int64``.
        place: Device placement.  Default is the current device.

    Returns:
        Array of Poisson-distributed random values.
    """
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if place is not None:
        kwargs["place"] = place
    return _poisson(lam, shape, **kwargs)
