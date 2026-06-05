"""Setup script for Insight7 Python bindings.

The C++ backend (.so files) must be built with CMake first:
    mkdir -p build && cd build
    cmake .. -DINSIGHT_BUILD_PYTHON_BINDING=ON
    make -j$(nproc) insight_python

Then install:
    pip install .
"""

from setuptools import setup, find_packages

setup(
    name="insight",
    version="1.0.0",
    description="Insight7 — lightweight scientific computing framework with GPU acceleration",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(where="bindings/python", include=["insight*"]),
    package_dir={"": "bindings/python"},
    package_data={"insight": ["*.so", "*.dll", "*.dylib"]},
    install_requires=["numpy>=1.20"],
)
