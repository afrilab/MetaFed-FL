"""
Setup script for MetaFed-FL package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_long_description():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "MetaFed-FL: Federated Learning for Metaverse Systems"

# Read requirements
def read_requirements(filename):
    try:
        with open(filename, "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

setup(
    name="metafed-fl",
    version="1.0.0",
    author="Muhammet Anil Yagiz, Zeynep Sude Cengiz, Polat Goktas",
    author_email="author@example.com",
    description="Federated Learning for Metaverse Infrastructures with MARL, Privacy, and Sustainability",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/MetaFed-FL",
    project_urls={
        "Bug Tracker": "https://github.com/username/MetaFed-FL/issues",
        "Documentation": "https://metafed-fl.readthedocs.io",
        "Source Code": "https://github.com/username/MetaFed-FL",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.2.2",
        "torchvision>=0.17.2",
        "numpy>=1.26.4",
        "pandas>=2.2.2",
        "matplotlib>=3.9.2",
        "timm>=1.0.8",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "scikit-learn>=1.1.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.19.0",
            "myst-parser>=0.18.0",
        ],
        "experiments": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.4.0",
            "ipywidgets>=8.0.0",
            "plotly>=5.10.0",
        ],
        "privacy": [
            "opacus>=1.4.0",
            "crypten>=0.4.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.19.0",
            "myst-parser>=0.18.0",
            "jupyter>=1.0.0",
            "jupyterlab>=3.4.0",
            "ipywidgets>=8.0.0",
            "plotly>=5.10.0",
            "opacus>=1.4.0",
            "crypten>=0.4.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "metafed-mnist=experiments.mnist.run_experiment:main",
            "metafed-cifar10=experiments.cifar10.run_experiment:main",
            "metafed-benchmark=scripts.run_benchmarks:main",
        ],
    },
    include_package_data=True,
    package_data={
        "metafed": [
            "configs/*.yaml",
            "configs/*.yml",
            "data/*.json",
        ],
    },
    zip_safe=False,
    keywords=[
        "federated learning",
        "machine learning",
        "distributed systems",
        "privacy preserving",
        "multi-agent reinforcement learning",
        "carbon aware computing",
        "sustainable AI",
        "metaverse",
        "PyTorch",
    ],
)