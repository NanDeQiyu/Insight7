"""Timer tests for Insight7 Python binding."""
import time
import insight as ins


def test_timer_cpu_create():
    t = ins.Timer((0, 0))
    assert t is not None
    assert hasattr(t, "start")
    assert hasattr(t, "stop")
    assert hasattr(t, "elapsed_ms")


def test_timer_cpu_measure():
    t = ins.Timer((0, 0))
    t.start()
    time.sleep(0.005)
    t.stop()
    ms = t.elapsed_ms()
    assert ms >= 0.0
    assert ms < 100.0  # Should be ~5ms


def test_timer_cpu_context_manager():
    with ins.Timer((0, 0)) as t:
        time.sleep(0.01)
    ms = t.elapsed_ms()
    assert ms >= 0.0


def test_timer_cpu_manual():
    t = ins.Timer((0, 0))
    t.start()
    time.sleep(0.003)
    t.stop()
    ms = t.elapsed_ms()
    assert ms >= 0.0


def test_timer_invalid_place():
    import pytest
    with pytest.raises(TypeError):
        ins.Timer("invalid")
    with pytest.raises(TypeError):
        ins.Timer(123)


def test_timer_gpu():
    if not ins.is_device_available("gpu"):
        pytest.skip("GPU not available")
    import insight as ins
    ins.load_backend("cuda")
    t = ins.Timer((1, 0))
    t.start()
    a = ins.ones([256, 256], dtype=ins.float32, place=ins.GPUPlace(0))
    b = ins.full([256, 256], 2.0, dtype=ins.float32, place=ins.GPUPlace(0))
    c = a + b
    t.stop()
    ms = t.elapsed_ms()
    assert ms >= 0.0
