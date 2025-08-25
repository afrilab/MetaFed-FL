"""
Federated learning algorithms.

This module contains implementations of various federated learning algorithms
including FedAvg, FedProx, and SCAFFOLD with their complete configurations.
"""

from .fedavg import FedAvgAlgorithm, create_fedavg_experiment
from .fedprox import FedProxAlgorithm, create_fedprox_experiment, tune_fedprox_mu
from .scaffold import SCAFFOLDAlgorithm, create_scaffold_experiment, compare_scaffold_vs_fedavg

__all__ = [
    "FedAvgAlgorithm",
    "FedProxAlgorithm", 
    "SCAFFOLDAlgorithm",
    "create_fedavg_experiment",
    "create_fedprox_experiment",
    "create_scaffold_experiment",
    "tune_fedprox_mu",
    "compare_scaffold_vs_fedavg"
]