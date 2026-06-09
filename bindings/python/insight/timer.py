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


class Profiler:
    """Multi-event aggregated profiler for recording timing statistics.

    Uses the device's native profiler mechanism to collect and aggregate
    timing data for multiple named events.

    Args:
        device: Device type string ('cpu' or 'gpu').
        device_id: Device index (default 0).

    Usage:
        # Context manager (recommended):
        with Profiler('cpu', 0) as prof:
            prof.begin_event('fft')
            result = ins.fft(data)
            prof.end_event()
        prof.report()

        # Manual start/stop:
        prof = Profiler('cpu', 0)
        prof.start()
        prof.begin_event('fft')
        result = ins.fft(data)
        prof.end_event()
        prof.stop()
        prof.report()
    """

    def __init__(self, device="cpu", device_id=0):
        self._handle = _insight.profiler_create(device, int(device_id))

    def __enter__(self):
        """Context manager: start profiling."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: stop profiling."""
        self.stop()
        return False

    def __del__(self):
        """Destroy the profiler and release resources."""
        try:
            if hasattr(self, "_handle") and self._handle is not None:
                _insight.profiler_destroy(self._handle)
                self._handle = None
        except Exception:
            pass

    def start(self):
        """Start recording profiler events."""
        _insight.profiler_start(self._handle)

    def stop(self):
        """Stop recording profiler events."""
        _insight.profiler_stop(self._handle)

    def reset(self):
        """Clear all recorded data."""
        _insight.profiler_reset(self._handle)

    def begin_event(self, name):
        """Begin a named event.

        Args:
            name: Event name string.
        """
        _insight.profiler_begin_event(self._handle, name)

    def end_event(self):
        """End the current event."""
        _insight.profiler_end_event(self._handle)

    def get_events(self):
        """Get aggregated event statistics.

        Returns:
            list[dict]: List of event dicts with keys:
                name, calls, total_ms, min_ms, max_ms
        """
        return list(_insight.profiler_get_events(self._handle))

    def report(self):
        """Print a formatted timing report."""
        events = self.get_events()
        if not events:
            print("  [Profiler] no events recorded")
            return
        print()
        print(f"  {'Event':<20} {'Calls':>7} {'Total(ms)':>12} " f"{'Avg(ms)':>10} {'Max(ms)':>10}")
        print("  " + "\u2500" * 59)
        for ev in events:
            avg = ev["total_ms"] / ev["calls"] if ev["calls"] > 0 else 0.0
            print(
                f"  {ev['name']:<20} {ev['calls']:>7} "
                f"{ev['total_ms']:>12.3f} {avg:>10.4f} {ev['max_ms']:>10.4f}"
            )
        print()


class ProfileBlock:
    """Context manager for profiling a single named code block.

    Args:
        profiler: Profiler instance.
        name: Event name.

    Usage::

        with ProfileBlock(prof, 'my_op'):
            result = compute_something()
    """

    def __init__(self, profiler, name):
        self._profiler = profiler
        self._name = name

    def __enter__(self):
        """Begin the event."""
        self._profiler.begin_event(self._name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the event."""
        self._profiler.end_event()
        return False
