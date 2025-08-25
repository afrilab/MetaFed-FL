"""
FedProx (Federated Optimization) Algorithm Implementation.

Paper: Federated Optimization in Heterogeneous Networks
Authors: Tian Li et al.
"""

from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn
from ..core.client import FedProxClient
from ..core.aggregation import FedProxAggregator
import logging

logger = logging.getLogger(__name__)


class FedProxAlgorithm:
    """
    FedProx algorithm implementation with proximal term.
    
    FedProx is designed to handle system heterogeneity and partial work
    by adding a proximal term to the local objective function.
    """
    
    def __init__(
        self,
        model_template: nn.Module,
        learning_rate: float = 0.01,
        local_epochs: int = 5,
        mu: float = 0.01,
        weighted_aggregation: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize FedProx algorithm.
        
        Args:
            model_template: Neural network model template
            learning_rate: Client learning rate
            local_epochs: Number of local training epochs
            mu: Proximal term coefficient
            weighted_aggregation: Whether to use weighted aggregation
            device: Device for computation
        """
        self.model_template = model_template
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.mu = mu
        self.device = device
        
        # Initialize aggregator
        self.aggregator = FedProxAggregator(weighted=weighted_aggregation)
        
        logger.info(f"Initialized FedProx algorithm with lr={learning_rate}, "
                   f"local_epochs={local_epochs}, mu={mu}")
    
    def create_client(
        self,
        client_id: int,
        train_loader: torch.utils.data.DataLoader
    ) -> FedProxClient:
        """
        Create a FedProx client.
        
        Args:
            client_id: Unique client identifier
            train_loader: Training data loader for this client
            
        Returns:
            Configured FedProx client
        """
        return FedProxClient(
            client_id=client_id,
            train_loader=train_loader,
            model_template=self.model_template,
            lr=self.learning_rate,
            device=self.device,
            local_epochs=self.local_epochs,
            mu=self.mu
        )
    
    def get_aggregator(self) -> FedProxAggregator:
        """Get the FedProx aggregator."""
        return self.aggregator
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information and hyperparameters."""
        return {
            "name": "FedProx",
            "learning_rate": self.learning_rate,
            "local_epochs": self.local_epochs,
            "mu": self.mu,
            "weighted_aggregation": self.aggregator.weighted,
            "paper": "Federated Optimization in Heterogeneous Networks",
            "year": 2020,
            "authors": "Tian Li et al.",
            "key_features": [
                "Proximal term for system heterogeneity",
                "Partial work tolerance",
                "Improved convergence in non-IID settings"
            ]
        }
    
    def validate_hyperparameters(self) -> bool:
        """Validate algorithm hyperparameters."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.local_epochs <= 0:
            raise ValueError("Local epochs must be positive")
        
        if self.mu < 0:
            raise ValueError("Proximal term coefficient (mu) must be non-negative")
        
        return True
    
    def get_recommended_mu(self, heterogeneity_level: str = "medium") -> float:
        """
        Get recommended mu value based on data heterogeneity.
        
        Args:
            heterogeneity_level: Level of data heterogeneity ("low", "medium", "high")
            
        Returns:
            Recommended mu value
        """
        recommendations = {
            "low": 0.001,      # Homogeneous data
            "medium": 0.01,    # Moderate non-IID
            "high": 0.1        # Highly non-IID
        }
        
        return recommendations.get(heterogeneity_level, 0.01)


def create_fedprox_experiment(
    model_template: nn.Module,
    train_datasets: List[torch.utils.data.Dataset],
    **kwargs
) -> Tuple[List[FedProxClient], FedProxAggregator]:
    """
    Create a complete FedProx experiment setup.
    
    Args:
        model_template: Neural network model
        train_datasets: List of training datasets for clients
        **kwargs: Additional hyperparameters
        
    Returns:
        Tuple of (clients, aggregator)
    """
    # Extract hyperparameters
    learning_rate = kwargs.get('learning_rate', 0.01)
    local_epochs = kwargs.get('local_epochs', 5)
    mu = kwargs.get('mu', 0.01)
    batch_size = kwargs.get('batch_size', 32)
    device = kwargs.get('device', 'cpu')
    weighted_aggregation = kwargs.get('weighted_aggregation', True)
    
    # Create algorithm instance
    algorithm = FedProxAlgorithm(
        model_template=model_template,
        learning_rate=learning_rate,
        local_epochs=local_epochs,
        mu=mu,
        weighted_aggregation=weighted_aggregation,
        device=device
    )
    
    # Create clients
    clients = []
    for i, dataset in enumerate(train_datasets):
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        client = algorithm.create_client(client_id=i, train_loader=train_loader)
        clients.append(client)
    
    aggregator = algorithm.get_aggregator()
    
    logger.info(f"Created FedProx experiment with {len(clients)} clients, mu={mu}")
    
    return clients, aggregator


def tune_fedprox_mu(
    model_template: nn.Module,
    validation_datasets: List[torch.utils.data.Dataset],
    mu_candidates: List[float] = [0.001, 0.01, 0.1, 1.0],
    **kwargs
) -> float:
    """
    Hyperparameter tuning for FedProx mu parameter.
    
    Args:
        model_template: Neural network model
        validation_datasets: Validation datasets for tuning
        mu_candidates: List of mu values to try
        **kwargs: Additional experiment parameters
        
    Returns:
        Best mu value based on validation performance
    """
    best_mu = mu_candidates[0]
    best_performance = float('-inf')
    
    logger.info(f"Tuning FedProx mu parameter with candidates: {mu_candidates}")
    
    for mu in mu_candidates:
        # Create FedProx experiment with current mu
        kwargs['mu'] = mu
        clients, aggregator = create_fedprox_experiment(
            model_template, validation_datasets, **kwargs
        )
        
        # Run short experiment for validation
        # This is a simplified version - in practice you'd run a full validation loop
        performance = _evaluate_mu_performance(clients, aggregator, mu)
        
        if performance > best_performance:
            best_performance = performance
            best_mu = mu
        
        logger.info(f"mu={mu}: performance={performance:.4f}")
    
    logger.info(f"Best mu: {best_mu} with performance: {best_performance:.4f}")
    return best_mu


def _evaluate_mu_performance(clients, aggregator, mu):
    """Simplified performance evaluation for mu tuning."""
    # This is a placeholder for actual performance evaluation
    # In practice, you would run a few rounds of training and evaluate
    import random
    return random.random()  # Mock performance score