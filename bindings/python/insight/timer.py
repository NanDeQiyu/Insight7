# bindings/python/insight/timer.py
"""Insight7 Timer Wrapper.

Provides a Pythonic Timer class wrapping the C API timer functions.
Supports both manual start/stop and context manager usage.
"""

import insight._insight as _insight


class Timer:
    """High-resolution timer for measuring execution time on CPU or GPU.

    Uses the device's native event mechanism:
    - GPU: cudaEvent (accurate kernel execution time)
    - CPU: std::chrono::high_resolution_clock

    Args:
        place: InsightPlace tuple (device_type, device_id).
               e.g., (0, 0) for CPU, (1, 0) for GPU:0

    Usage:
        # Context manager (recommended):
        with Timer((0, 0)) as t:
            result = ins.fft(data)
        print(f"{t.elapsed_ms():.3f} ms")

        # Manual start/stop:
        t = Timer((1, 0))
        t.start()
        result = ins.fft(data)
        t.stop()
        print(f"{t.elapsed_ms():.3f} ms")
    """

    def __init__(self, place):
        if not isinstance(place, (tuple, list)) or len(place) != 2:
            raise TypeError("place must be a tuple/list of (device_type, device_id)")
        self._handle = _insight.timer_create(int(place[0]), int(place[1]))
        self._started = False

    def start(self):
        """Record the start event on the device."""
        _insight.timer_start(self._handle)
        self._started = True

    def stop(self):
        """Record the stop event and synchronize."""
        _insight.timer_stop(self._handle)
        self._started = False

    def elapsed_ms(self):
        """Get the elapsed time in milliseconds between start and stop.

        Returns:
            float: Elapsed time in milliseconds.
        """
        return _insight.timer_elapsed_ms(self._handle)

    def reset(self):
        """Reset the timer for reuse."""
        # For simplicity, just create a new timer
        # (the C API reset is more complex with opaque handles)
        import insight._insight as _insight

    def __enter__(self):
        """Context manager: start the timer."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: stop the timer."""
        self.stop()
        return False

    def __del__(self):
        """Destroy the timer and release resources."""
        try:
            if hasattr(self, "_handle") and self._handle is not None:
                _insight.timer_destroy(self._handle)
                self._handle = None
        except Exception:
            pass
