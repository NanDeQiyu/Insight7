"""Linear algebra operations.

Provides matrix multiplication, decompositions, eigenvalue solvers,
and other standard linear algebra routines.

All functions operate on :class:`insight.Array` and return results
as :class:`insight.Array` (or tuples thereof).
"""

from insight._insight import (
    matmul as _matmul,
    dot as _dot,
    inner as _inner,
    outer as _outer,
    det as _det,
    slogdet as _slogdet,
    inv as _inv,
    pinv as _pinv,
    solve as _solve,
    lstsq as _lstsq,
    svd as _svd,
    svdvals as _svdvals,
    eigh as _eigh,
    eigvalsh as _eigvalsh,
    eig as _eig,
    eigvals as _eigvals,
    cholesky as _cholesky,
    qr as _qr,
    lq as _lq,
    lu as _lu,
    norm as _norm,
    cond as _cond,
    matrix_rank as _matrix_rank,
    matrix_power as _matrix_power,
    trace as _trace,
    cov as _cov,
)

__all__ = [
    "matmul",
    "dot",
    "inner",
    "outer",
    "det",
    "slogdet",
    "inv",
    "pinv",
    "solve",
    "lstsq",
    "svd",
    "svdvals",
    "eigh",
    "eigvalsh",
    "eig",
    "eigvals",
    "cholesky",
    "qr",
    "lq",
    "lu",
    "norm",
    "cond",
    "matrix_rank",
    "matrix_power",
    "trace",
    "cov",
]


def matmul(a, b):
    """Matrix multiplication of two arrays.

    Args:
        a: Left operand (at least 1-D).
        b: Right operand (at least 1-D).

    Returns:
        Matrix product of *a* and *b*.
    """
    return _matmul(a, b)


def dot(a, b):
    """Dot product of two arrays.

    For 1-D arrays this is the inner product.  For 2-D arrays it is
    equivalent to :func:`matmul`.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        Dot product result.
    """
    return _dot(a, b)


def inner(a, b):
    """Inner product of two arrays.

    Args:
        a: First operand (1-D or higher).
        b: Second operand (1-D or higher, same shape as *a*).

    Returns:
        Inner product array.
    """
    return _inner(a, b)


def outer(a, b):
    """Outer product of two 1-D arrays.

    Args:
        a: First 1-D array of size M.
        b: Second 1-D array of size N.

    Returns:
        2-D array of shape (M, N).
    """
    return _outer(a, b)


def det(x):
    """Compute the determinant of a square matrix.

    Args:
        x: Input square matrix (2-D array).

    Returns:
        Scalar array containing the determinant.
    """
    return _det(x)


def slogdet(x):
    """Compute the sign and (natural) log of the determinant.

    Useful when the determinant itself may overflow or underflow.

    Args:
        x: Input square matrix (2-D array).

    Returns:
        Tuple ``(sign, logabsdet)`` where *sign* is -1, 0, or 1 and
        *logabsdet* is the log of the absolute determinant.
    """
    return _slogdet(x)


def inv(x):
    """Compute the inverse of a square matrix.

    Args:
        x: Input square matrix (2-D array).

    Returns:
        Inverse of *x*.
    """
    return _inv(x)


def pinv(x, rcond=-1.0):
    """Compute the Moore-Penrose pseudo-inverse of a matrix.

    Args:
        x: Input matrix (2-D array).
        rcond: Cutoff for small singular values.  Singular values smaller
            than ``rcond * max(singular_values)`` are set to zero.
            Default is -1.0 (use machine precision).

    Returns:
        Pseudo-inverse of *x*.
    """
    return _pinv(x, rcond=rcond)


def solve(a, b):
    """Solve a linear system ``A x = b``.

    Args:
        a: Coefficient matrix (2-D square array).
        b: Right-hand side (1-D or 2-D array).

    Returns:
        Solution array *x*.
    """
    return _solve(a, b)


def lstsq(a, b, rcond=-1.0):
    """Compute the least-squares solution to ``A x = b``.

    Args:
        a: Coefficient matrix (2-D array, M x N).
        b: Right-hand side (1-D or 2-D array, length M).
        rcond: Cutoff for small singular values.  Default is -1.0.

    Returns:
        Tuple ``(x, residuals, rank, sv)`` of the solution, residuals,
        effective rank, and singular values.
    """
    return _lstsq(a, b, rcond=rcond)


def svd(x, full_matrices=True):
    """Singular Value Decomposition.

    Factors *x* as ``U @ diag(S) @ Vh``.

    Args:
        x: Input matrix (2-D array).
        full_matrices: If True (default), return full *U* and *Vh*
            matrices.  If False, return compact forms.

    Returns:
        Tuple ``(U, S, Vh)``.
    """
    return _svd(x, full_matrices=full_matrices)


def svdvals(x):
    """Compute the singular values of a matrix.

    Args:
        x: Input matrix (2-D array).

    Returns:
        1-D array of singular values in descending order.
    """
    return _svdvals(x)


def eigh(x, uplo="L"):
    """Eigenvalue decomposition for symmetric or Hermitian matrices.

    Args:
        x: Input symmetric/Hermitian matrix (2-D array).
        uplo: ``'L'`` (default) uses the lower triangle; ``'U'`` uses
            the upper triangle.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)``.
    """
    return _eigh(x, uplo=uplo)


def eigvalsh(x, uplo="L"):
    """Eigenvalues of a symmetric or Hermitian matrix.

    Args:
        x: Input symmetric/Hermitian matrix (2-D array).
        uplo: ``'L'`` (default) uses the lower triangle; ``'U'`` uses
            the upper triangle.

    Returns:
        1-D array of eigenvalues in ascending order.
    """
    return _eigvalsh(x, uplo=uplo)


def eig(x):
    """Eigenvalue decomposition of a general square matrix.

    Args:
        x: Input square matrix (2-D array).

    Returns:
        Tuple ``(eigenvalues, eigenvectors)``.
    """
    return _eig(x)


def eigvals(x):
    """Compute the eigenvalues of a general square matrix.

    Args:
        x: Input square matrix (2-D array).

    Returns:
        1-D array of eigenvalues.
    """
    return _eigvals(x)


def cholesky(x, lower=True):
    """Cholesky decomposition of a positive-definite matrix.

    Args:
        x: Input symmetric positive-definite matrix (2-D array).
        lower: If True (default), return the lower-triangular factor *L*
            such that ``x = L @ L.T``.  If False, return upper *U*.

    Returns:
        Lower (or upper) Cholesky factor.
    """
    return _cholesky(x, lower=lower)


def qr(x, mode="reduced"):
    """QR decomposition.

    Args:
        x: Input matrix (2-D array).
        mode: ``'reduced'`` (default) returns compact Q and R;
            ``'full'`` returns full Q and R.

    Returns:
        Tuple ``(Q, R)``.
    """
    return _qr(x, mode=mode)


def lq(x, mode="reduced"):
    """LQ decomposition.

    Args:
        x: Input matrix (2-D array).
        mode: ``'reduced'`` (default) returns compact L and Q;
            ``'full'`` returns full L and Q.

    Returns:
        Tuple ``(L, Q)``.
    """
    return _lq(x, mode=mode)


def lu(x, pivot=True):
    """LU decomposition with partial pivoting.

    Args:
        x: Input matrix (2-D array).
        pivot: If True (default), use partial pivoting.

    Returns:
        Tuple ``(P, L, U)`` where ``P @ L @ U == x``.
    """
    return _lu(x, pivot=pivot)


def norm(x, ord=2.0):
    """Compute the matrix or vector norm.

    Args:
        x: Input array.
        ord: Order of the norm.  Common values: 1, 2, inf, 'fro'.

    Returns:
        Scalar array containing the norm.
    """
    return _norm(x, ord=ord)


def cond(x, p=2.0):
    """Compute the condition number of a matrix.

    Args:
        x: Input matrix (2-D array).
        p: Order of the norm.  Default is 2.

    Returns:
        Scalar array containing the condition number.
    """
    return _cond(x, p=p)


def matrix_rank(x, tol=-1.0):
    """Compute the rank of a matrix using SVD.

    Args:
        x: Input matrix (2-D array).
        tol: Threshold below which singular values are considered zero.
            Default is -1.0 (use machine-precision-based threshold).

    Returns:
        Scalar array containing the effective rank.
    """
    return _matrix_rank(x, tol=tol)


def matrix_power(x, n):
    """Raise a square matrix to an integer power.

    Args:
        x: Input square matrix (2-D array).
        n: Non-negative integer exponent.

    Returns:
        Matrix raised to the *n*-th power.
    """
    return _matrix_power(x, n)


def trace(x):
    """Compute the trace (sum of diagonal elements) of a matrix.

    Args:
        x: Input matrix (2-D array).

    Returns:
        Scalar array containing the trace.
    """
    return _trace(x)


def cov(x, rowvar=True, ddof=1):
    """Compute the covariance matrix.

    Args:
        x: Input array.  Each row (if *rowvar* is True) represents a
            variable, each column an observation.
        rowvar: If True (default), each row is a variable.
        ddof: Delta degrees of freedom.  Default is 1.

    Returns:
        Covariance matrix.
    """
    return _cov(x, rowvar=rowvar, ddof=ddof)
