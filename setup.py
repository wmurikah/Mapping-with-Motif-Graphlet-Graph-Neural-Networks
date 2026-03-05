"""Setup script for MGI-GNN package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mgi-gnn",
    version="1.0.0",
    author="Kelvin Munene Nyaga, Faith Mueni Musyoka, David Muchangi Mugo, Wilberforce Murikah",
    author_email="kelvinmunene.n@gmail.com",
    description=(
        "Motif-Graphlet Integrated Graph Neural Network for "
        "Competency Mapping in Educational Assessment"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wmurikah/Motif-Graphlet-Graph-Neural-Networks",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "networkx>=3.1",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "pyg": ["torch-geometric>=2.4.0", "torch-scatter>=2.1.0", "torch-sparse>=0.6.18"],
        "xgboost": ["xgboost>=1.7.0"],
        "wandb": ["wandb>=0.15.0"],
        "full": [
            "torch-geometric>=2.4.0",
            "xgboost>=1.7.0",
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
            "seaborn>=0.12.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="graph-neural-networks education assessment motifs graphlets",
)
