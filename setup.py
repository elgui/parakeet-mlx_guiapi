#!/usr/bin/env python3
"""
Setup script for Parakeet-MLX GUI and API.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="parakeet_mlx_guiapi",
    version="0.1.0",
    author="MLX Community",
    author_email="info@example.com",
    description="GUI and API for Parakeet-MLX speech-to-text library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/parakeet-mlx_guiapi",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "parakeet-server=run:main",
            "parakeet-client=client:main",
        ],
    },
)
