"""
Setup script for the Hierarchical Predictive Coding Network package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hpcn",
    version="0.1.0",
    author="Raktim Mondol",
    author_email="raktim.live@gmail.com",
    description="A PyTorch implementation of Hierarchical Predictive Coding Networks inspired by neuroscience",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raktim-mondol/hierarchical-predictive-coding-network",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.60.0",
        "scikit-learn>=0.24.0",
    ],
)