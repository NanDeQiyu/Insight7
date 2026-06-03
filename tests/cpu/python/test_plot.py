"""Plot CPU binding tests — 13 tests aligned with C++ plot test suite.

Tests that plot functions exist and can be called without crashing.
Plotting is hard to verify numerically, so we only check smoke behavior.

Functions tested: plot, scatter, bar, hist, imshow, contour,
                  subplot, title, xlabel, ylabel, legend, savefig, close

Run with:
    PYTHONPATH=bindings/python:build/bindings/python python -m pytest tests/cpu/python/test_plot.py -v
"""

import sys
import os
import subprocess
import tempfile
import pytest

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, os.path.join(_root, "bindings", "python"))
sys.path.insert(0, os.path.join(_root, "build", "bindings", "python"))

try:
    import insight as ins
    import numpy as np
except ImportError:
    pytest.skip("Insight or NumPy not available", allow_module_level=True)

# Skip if plot module not compiled (requires INSIGHT_USE_MATPLOT=ON)
try:
    _plt = ins.plot
except (AttributeError, ImportError):
    pytest.skip("ins.plot not available (INSIGHT_USE_MATPLOT not enabled)", allow_module_level=True)

# Skip if gnuplot binary is not available (matplotplusplus needs it for rendering)
import subprocess as _sp

try:
    _gnuplot_check = _sp.run(["gnuplot", "--version"], capture_output=True, timeout=5)
    if _gnuplot_check.returncode != 0:
        pytest.skip("gnuplot not installed, skipping plot tests", allow_module_level=True)
except (FileNotFoundError, _sp.TimeoutExpired):
    pytest.skip("gnuplot not installed, skipping plot tests", allow_module_level=True)


class TestPlotCPU:
    """Plot binding smoke tests — tests 1-13."""

    def test_plot_basic(self):
        """Test 1: plot y-data without crashing."""
        y = ins.from_numpy(np.array([1.0, 3.0, 2.0, 4.0]))
        ins.plot.plot(y)
        ins.plot.clf()

    def test_scatter_basic(self):
        """Test 2: scatter without crashing."""
        x = ins.from_numpy(np.array([1.0, 2.0, 3.0]))
        y = ins.from_numpy(np.array([4.0, 5.0, 6.0]))
        ins.plot.scatter(x, y)
        ins.plot.clf()

    def test_bar_basic(self):
        """Test 3: bar chart without crashing."""
        y = ins.from_numpy(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))
        ins.plot.bar(y)
        ins.plot.clf()

    def test_hist_basic(self):
        """Test 4: histogram without crashing."""
        data = ins.from_numpy(np.random.randn(100))
        ins.plot.hist(data, bins=10)
        ins.plot.clf()

    def test_imshow_basic(self):
        """Test 5: imshow without crashing."""
        data = ins.from_numpy(np.random.rand(5, 5))
        ins.plot.imshow(data)
        ins.plot.clf()

    def test_contour_basic(self):
        """Test 6: contour plot without crashing."""
        x_np = np.linspace(-2, 2, 10, dtype=np.float64)
        y_np = np.linspace(-2, 2, 10, dtype=np.float64)
        xx, yy = np.meshgrid(x_np, y_np)
        zz = np.sin(xx) * np.cos(yy)
        X = ins.from_numpy(xx)
        Y = ins.from_numpy(yy)
        Z = ins.from_numpy(zz)
        ins.plot.contour(X, Y, Z)
        ins.plot.clf()

    def test_subplot_basic(self):
        """Test 7: subplot without crashing."""
        ins.plot.subplot(2, 1, 1)
        ins.plot.clf()

    def test_title_basic(self):
        """Test 8: title without crashing."""
        ins.plot.title("Test Title")
        ins.plot.clf()

    def test_xlabel_basic(self):
        """Test 9: xlabel without crashing."""
        ins.plot.xlabel("X Axis")
        ins.plot.clf()

    def test_ylabel_basic(self):
        """Test 10: ylabel without crashing."""
        ins.plot.ylabel("Y Axis")
        ins.plot.clf()

    def test_legend_basic(self):
        """Test 11: legend without crashing."""
        y = ins.from_numpy(np.array([1.0, 2.0, 3.0]))
        ins.plot.plot(y)
        ins.plot.legend(["data"])
        ins.plot.clf()

    def test_savefig_basic(self):
        """Test 12: save figure to file without crashing."""
        y = ins.from_numpy(np.array([1.0, 2.0, 3.0]))
        ins.plot.plot(y)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmpfile = f.name
        try:
            ins.plot.save(tmpfile)
            assert os.path.exists(tmpfile)
        finally:
            ins.plot.clf()
            if os.path.exists(tmpfile):
                os.unlink(tmpfile)

    def test_close_basic(self):
        """Test 13: clf (clear figure) without crashing."""
        ins.plot.figure(1)
        ins.plot.clf()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
