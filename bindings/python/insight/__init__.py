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

    # Signal processing
    w = ins.signal.hann(256)
    f, Pxx = ins.signal.welch(x, fs=1000)

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
        DType,
        Place,
        Shape,
        Slice,
        Array,
        init,
        is_initialized,
        load_backend,
        # Creation
        zeros,
        ones,
        full,
        eye,
        arange,
        linspace,
        logspace,
        from_numpy,
        from_array,
        zeros_like,
        ones_like,
        # Elementwise binary
        add,
        sub,
        mul,
        div,
        pow,
        mod,
        maximum,
        minimum,
        equal,
        not_equal,
        greater,
        less,
        greater_equal,
        less_equal,
        logical_and,
        logical_or,
        logical_xor,
        logical_not,
        # Unary math
        abs,
        negative,
        square,
        sqrt,
        exp,
        log,
        log2,
        log10,
        sin,
        cos,
        tan,
        asin,
        acos,
        atan,
        sinh,
        cosh,
        tanh,
        floor,
        ceil,
        round,
        sign,
        isnan,
        isinf,
        isfinite,
        where,
        # Reduction
        sum,
        mean,
        max,
        min,
        prod,
        argmax,
        argmin,
        any,
        all,
        var,
        std,
        cumsum,
        cumprod,
        # Manipulation
        concat,
        stack,
        vstack,
        hstack,
        split,
        tile,
        repeat,
        flip,
        pad,
        reshape,
        flatten,
        ravel,
        squeeze,
        unsqueeze,
        roll,
        # Linalg
        matmul,
        dot,
        inner,
        outer,
        det,
        slogdet,
        inv,
        pinv,
        solve,
        lstsq,
        svd,
        svdvals,
        eigh,
        eigvalsh,
        eig,
        eigvals,
        cholesky,
        qr,
        lq,
        lu,
        norm,
        cond,
        matrix_rank,
        matrix_power,
        trace,
        cov,
        # FFT
        fft,
        ifft,
        fft2,
        ifft2,
        fftn,
        ifftn,
        rfft,
        irfft,
        fftfreq,
        rfftfreq,
        # Signal (top-level aliases)
        convolve,
        unwrap,
        sinc,
        # Random
        rand,
        randn,
        randint,
        normal,
        uniform,
        randperm,
        # Cast & Indexing
        cast,
        take,
        nonzero,
        flatnonzero,
        argsort,
        sort,
        masked_select,
        searchsorted,
        # Place constructors
        CPUPlace,
        GPUPlace,
    )

    # Signal submodule (ins.signal.*) — uses wrapper layer
    from . import signal as _signal_module

    signal = _signal_module

    # Import result types from native binding (not wrapped in submodule)
    from ._insight.signal import (
        SpectralResult,
        SpectrogramResult,
        GaussPulseResult,
        ChirpMethod,
    )

    # Re-export signal functions at top level for convenience
    from .signal import (  # noqa: F401
        # Windows
        general_cosine,
        get_window,
        boxcar,
        triang,
        parzen,
        bohman,
        bartlett,
        cosine,
        blackman,
        nuttall,
        blackmanharris,
        flattop,
        hann,
        general_hamming,
        hamming,
        tukey,
        barthann,
        kaiser,
        gaussian,
        general_gaussian,
        chebwin,
        taylor,
        # Waveforms
        sawtooth,
        square_wf,
        gausspulse,
        gausspulse_full,
        chirp,
        unit_impulse,
        # B-Splines
        gauss_spline,
        cubic,
        quadratic,
        # Filter Design
        kaiser_beta,
        kaiser_atten,
        firwin,
        firwin2,
        cmplx_sort,
        # Convolution
        fftconvolve,
        correlate,
        convolve2d,
        correlate2d,
        choose_conv_method,
        correlation_lags,
        # Filtering
        hilbert,
        hilbert2,
        detrend,
        wiener,
        firfilter,
        lfilter,
        lfilter_zi,
        filtfilt,
        decimate,
        resample,
        resample_poly,
        freq_shift,
        # Spectral Analysis
        csd,
        welch,
        periodogram,
        coherence,
        spectrogram,
        stft,
        vectorstrength,
        # Wavelets
        morlet,
        ricker,
        morlet2,
        cwt,
        # Acoustics
        mel2hz,
        hz2mel,
        mel_frequencies,
        hz2bark,
        bark2hz,
        # Demod
        fm_demod,
        # Peak Finding
        argrelextrema,
        argrelmax,
        argrelmin,
        # Radar
        pulse_compression,
        pulse_doppler,
        cfar_alpha,
        ca_cfar,
        mvdr,
        ambgfun,
        # Estimation
        KalmanFilter,
        # Signal I/O
        read_bin,
        write_bin,
        unpack_bin,
        pack_bin,
        read_sigmf,
        write_sigmf,
    )

    # Plot submodule (ins.plot.*) — may not exist if INSIGHT_USE_MATPLOT=OFF
    try:
        from ._insight import plot as _plot_module

        plot = _plot_module
    except ImportError:
        pass

except ImportError as e:
    raise ImportError(
        f"Failed to load Insight native module: {e}\n"
        "Make sure you have built the Python binding:\n"
        "  cmake --build build --target insight_python"
    ) from e

__version__ = "1.0.0"
__all__ = [
    # Types
    "DType",
    "Place",
    "Shape",
    "Slice",
    "Array",
    # Init
    "init",
    "is_initialized",
    "load_backend",
    # Place
    "CPUPlace",
    "GPUPlace",
    # Dtypes
    "float32",
    "float64",
    "float16",
    "bfloat16",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool",
    "complex64",
    "complex128",
    # Creation
    "zeros",
    "ones",
    "full",
    "eye",
    "arange",
    "linspace",
    "from_numpy",
    "zeros_like",
    "ones_like",
    # Elementwise
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "mod",
    "maximum",
    "minimum",
    "equal",
    "not_equal",
    "greater",
    "less",
    "greater_equal",
    "less_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    # Unary
    "abs",
    "negative",
    "square",
    "sqrt",
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "floor",
    "ceil",
    "round",
    "sign",
    "isnan",
    "isinf",
    "isfinite",
    "where",
    # Reduction
    "sum",
    "mean",
    "max",
    "min",
    "prod",
    "argmax",
    "argmin",
    "any",
    "all",
    "var",
    "std",
    # Manipulation
    "concat",
    "stack",
    "split",
    "tile",
    "repeat",
    "flip",
    "pad",
    "reshape",
    "flatten",
    "ravel",
    "squeeze",
    "unsqueeze",
    "roll",
    # Linalg
    "matmul",
    "dot",
    "outer",
    "det",
    "inv",
    "solve",
    "svd",
    "eigh",
    "cholesky",
    "qr",
    "norm",
    "trace",
    # FFT
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "fftfreq",
    # Signal (top-level)
    "convolve",
    "unwrap",
    "sinc",
    # Signal submodule
    "signal",
    # Signal functions (top-level re-exports)
    "general_cosine",
    "get_window",
    "boxcar",
    "triang",
    "parzen",
    "bohman",
    "bartlett",
    "cosine",
    "exponential",
    "blackman",
    "nuttall",
    "blackmanharris",
    "flattop",
    "hann",
    "general_hamming",
    "hamming",
    "tukey",
    "barthann",
    "kaiser",
    "gaussian",
    "general_gaussian",
    "chebwin",
    "taylor",
    "sawtooth",
    "square_wf",
    "gausspulse",
    "gausspulse_full",
    "chirp",
    "unit_impulse",
    "ChirpMethod",
    "gauss_spline",
    "cubic",
    "quadratic",
    "kaiser_beta",
    "kaiser_atten",
    "firwin",
    "firwin2",
    "cmplx_sort",
    "fftconvolve",
    "correlate",
    "convolve2d",
    "correlate2d",
    "choose_conv_method",
    "correlation_lags",
    "hilbert",
    "hilbert2",
    "detrend",
    "wiener",
    "firfilter",
    "lfilter",
    "lfilter_zi",
    "filtfilt",
    "decimate",
    "resample",
    "resample_poly",
    "freq_shift",
    "SpectralResult",
    "SpectrogramResult",
    "GaussPulseResult",
    "csd",
    "welch",
    "periodogram",
    "coherence",
    "spectrogram",
    "stft",
    "vectorstrength",
    "morlet",
    "ricker",
    "morlet2",
    "cwt",
    "mel2hz",
    "hz2mel",
    "mel_frequencies",
    "hz2bark",
    "bark2hz",
    # Random
    "rand",
    "randn",
    "randint",
    "normal",
    "uniform",
    # Cast & Indexing
    "cast",
    "take",
    "nonzero",
    "argsort",
    "sort",
    # Plot submodule
    "plot",
]
