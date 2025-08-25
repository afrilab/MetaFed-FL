"""
Core federated learning components.

This module contains the fundamental building blocks for federated learning:
- Client implementations
- Server implementations  
- Aggregation algorithms
"""

from .client import Client, FedProxClient, SCAFFOLDClient
from .server import FederatedServer
from .aggregation import FedAvgAggregator, FedProxAggregator, SCAFFOLDAggregator

__all__ = [
    "Client",
    "FedProxClient", 
    "SCAFFOLDClient",
    "FederatedServer",
    "FedAvgAggregator",
    "FedProxAggregator",
    "SCAFFOLDAggregator"
]