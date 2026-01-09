"""Gaussian Splatting package

A Python package for generating 3D Gaussian Splatting models from a single image using depth estimation.
"""

from .gaussian_model import GaussianModel
from .renderer import render, Camera
from .trainer import Trainer

__all__ = [
    "GaussianModel",
    "render",
    "Camera",
    "Trainer",
]

__version__ = "0.1.0"
