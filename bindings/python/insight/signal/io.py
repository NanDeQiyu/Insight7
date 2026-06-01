"""Binary signal I/O functions.

Provides reading and writing of binary signal files and SigMF-format
data files.
"""

from insight._insight import DType as _DType
from insight._insight import signal as _signal

__all__ = [
    "read_bin",
    "write_bin",
    "pack_bin",
    "unpack_bin",
    "read_sigmf",
    "write_sigmf",
]

# Map common dtype strings to Insight DType enum values.
_DTYPE_MAP = {
    "int8": _DType.int8,
    "uint8": _DType.uint8,
    "int16": _DType.int16,
    "uint16": _DType.uint16,
    "int32": _DType.int32,
    "uint32": _DType.uint32,
    "int64": _DType.int64,
    "uint64": _DType.uint64,
    "float32": _DType.float32,
    "float64": _DType.float64,
    "complex64": _DType.complex64,
    "complex128": _DType.complex128,
}


def _resolve_dtype(dtype):
    """Resolve a dtype argument to a DType enum value.

    Args:
        dtype: A DType enum value, a string like ``'float64'``, or a
            numpy-style dtype.

    Returns:
        Corresponding DType enum value.
    """
    if isinstance(dtype, str):
        return _DTYPE_MAP.get(dtype, _DType.float64)
    return dtype


def read_bin(filename, dtype="float64"):
    """Read a binary file into an array.

    Args:
        filename: Path to the binary file.
        dtype: Data type for interpreting the raw bytes.  Can be a string
            (e.g., ``'float64'``, ``'int16'``) or a ``DType`` enum value.
            Default is ``'float64'``.

    Returns:
        Array containing the file contents interpreted as *dtype*.
    """
    return _signal.read_bin(filename, dtype=_resolve_dtype(dtype))


def write_bin(filename, data):
    """Write an array to a binary file.

    Args:
        filename: Output file path.
        data: Array to write.
    """
    _signal.write_bin(filename, data)


def pack_bin(filename, *arrays):
    """Pack one or more arrays into a binary file.

    Each array is serialised to raw bytes and concatenated into the
    output file.

    Args:
        filename: Output file path.
        *arrays: One or more arrays to pack.
    """
    for arr in arrays:
        packed = _signal.pack_bin(arr)
        _signal.write_bin(filename, packed)


def unpack_bin(filename, count):
    """Read and unpack a binary file into an array.

    Reads the raw bytes from *filename* and interprets them as
    ``float64`` samples.

    Args:
        filename: Path to the binary file.
        count: Number of samples to read.  If 0, reads the entire file.

    Returns:
        Array of unpacked samples.
    """
    raw = _signal.read_bin(filename, dtype=_DType.uint8, num_samples=count if count > 0 else 0)
    return _signal.unpack_bin(raw, _DType.float64)


def read_sigmf(filename):
    """Read a SigMF data file.

    Parses the companion ``.sigmf-meta`` JSON file (or a file derived
    from *filename*) to determine the data type and endianness, then
    reads the corresponding ``.sigmf-data`` file.

    Args:
        filename: Path to the ``.sigmf-data`` file (or the base name
            without extension, in which case ``.sigmf-data`` is appended).

    Returns:
        Array containing the parsed data.
    """
    return _signal.read_sigmf(filename)


def write_sigmf(filename, data, metadata=None):
    """Write an array in SigMF format.

    Packs the data and writes it to a ``.sigmf-data`` file.  If
    *metadata* is provided, it is written to the companion
    ``.sigmf-meta`` JSON file.

    Args:
        filename: Output ``.sigmf-data`` file path.
        data: Array to write.
        metadata: Optional dictionary of SigMF metadata.  If provided,
            it is serialised as JSON to the companion ``.sigmf-meta``
            file.
    """
    _signal.write_sigmf(filename, data)
    if metadata is not None:
        import json

        meta_file = filename.replace(".sigmf-data", ".sigmf-meta")
        if meta_file == filename:
            meta_file = filename + ".sigmf-meta"
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=2)
