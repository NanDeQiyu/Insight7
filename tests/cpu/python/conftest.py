"""conftest.py for CPU tests — force CPU device for all tests.

With PaddlePaddle-style default device (GPU when available), CPU tests
must explicitly set device to CPU to avoid GPU allocation failures.
"""

import sys
import os

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, os.path.join(_root, "bindings", "python"))
sys.path.insert(0, os.path.join(_root, "build", "bindings", "python"))

import insight as ins

# Force CPU device for all CPU tests (module-level, runs once at import)
ins.set_device(ins.CPUPlace())
