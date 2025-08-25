"""
Privacy-preserving mechanisms for federated learning.

This module provides implementations of various privacy-preserving techniques
including differential privacy and homomorphic encryption.
"""

from .differential_privacy import (
    DifferentialPrivacy,
    PrivacyAccountant,
    FederatedDifferentialPrivacy,
    create_dp_federated_experiment
)
from .homomorphic_encryption import (
    SecureAggregator,
    SecureFederatedAggregation,
    create_secure_federated_experiment
)

__all__ = [
    "DifferentialPrivacy",
    "PrivacyAccountant",
    "FederatedDifferentialPrivacy",
    "create_dp_federated_experiment",
    "SecureAggregator",
    "SecureFederatedAggregation",
    "create_secure_federated_experiment"
]