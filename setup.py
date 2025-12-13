#!/usr/bin/env python3
"""
Setup script for PGTP-Net
Pose-Guided Trajectory Prediction Network
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="pgtpnet",
    version="1.0.0",
    author="Sai Jayakar Vanam, Rithvik Shivva, Umar Hassan Makki Mohammad, Priyanka Lakariya",
    author_email="vanam.sai@northeastern.edu",
    description="Pose-Guided Trajectory Prediction Network for pedestrian motion forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/PGTP-Net",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pgtp-train=train:main",
            "pgtp-eval=evaluate:main",
            "pgtp-demo=demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
