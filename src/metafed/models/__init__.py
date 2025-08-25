"""
Neural network models for MetaFed-FL.
"""

from .simple_cnn import SimpleCNN
from .resnet import ResNet10t

__all__ = ["SimpleCNN", "ResNet10t"]