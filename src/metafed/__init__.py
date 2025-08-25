"""
MetaFed-FL: Federated Learning for Metaverse Systems

A comprehensive federated learning framework that integrates:
- Multi-Agent Reinforcement Learning (MARL) for dynamic client orchestration
- Privacy-preserving techniques (homomorphic encryption, differential privacy)
- Carbon-aware scheduling for sustainable resource management

Authors: Muhammet Anil Yagiz, Zeynep Sude Cengiz, Polat Goktas
"""

__version__ = "1.0.0"
__author__ = "Muhammet Anil Yagiz, Zeynep Sude Cengiz, Polat Goktas"
__email__ = "author@example.com"
__license__ = "MIT"

from . import core, algorithms, orchestration, privacy, green, models, data, utils

__all__ = [
    "core",
    "algorithms", 
    "orchestration",
    "privacy",
    "green",
    "models",
    "data",
    "utils"
]