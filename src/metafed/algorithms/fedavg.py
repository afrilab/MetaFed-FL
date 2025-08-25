"""
FedAvg (Federated Averaging) Algorithm Implementation.

Paper: Communication-Efficient Learning of Deep Networks from Decentralized Data
Authors: H. Brendan McMahan et al.
"""

from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn
from ..core.client import Client
from ..core.aggregation import FedAvgAggregator
import logging

logger = logging.getLogger(__name__)


class FedAvgAlgorithm:
    """
    FedAvg algorithm implementation with coordinated client and server components.
    
    This class provides a high-level interface for running FedAvg experiments
    with proper coordination between clients and aggregation.
    """
    
    def __init__(
        self,
        model_template: nn.Module,
        learning_rate: float = 0.01,
        local_epochs: int = 5,
        weighted_aggregation: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize FedAvg algorithm.
        
        Args:
            model_template: Neural network model template
            learning_rate: Client learning rate
            local_epochs: Number of local training epochs
            weighted_aggregation: Whether to use weighted aggregation
            device: Device for computation
        """
        self.model_template = model_template
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.device = device
        
        # Initialize aggregator
        self.aggregator = FedAvgAggregator(weighted=weighted_aggregation)
        
        logger.info(f"Initialized FedAvg algorithm with lr={learning_rate}, "
                   f"local_epochs={local_epochs}, weighted={weighted_aggregation}")
    
    def create_client(
        self,
        client_id: int,
        train_loader: torch.utils.data.DataLoader
    ) -> Client:
        """
        Create a FedAvg client.
        
        Args:
            client_id: Unique client identifier
            train_loader: Training data loader for this client
            
        Returns:
            Configured FedAvg client
        """
        return Client(
            client_id=client_id,
            train_loader=train_loader,
            model_template=self.model_template,
            lr=self.learning_rate,
            device=self.device,
            local_epochs=self.local_epochs
        )
    
    def get_aggregator(self) -> FedAvgAggregator:
        """Get the FedAvg aggregator."""
        return self.aggregator
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information and hyperparameters."""
        return {
            "name": "FedAvg",
            "learning_rate": self.learning_rate,
            "local_epochs": self.local_epochs,
            "weighted_aggregation": self.aggregator.weighted,
            "paper": "Communication-Efficient Learning of Deep Networks from Decentralized Data",
            "year": 2017,
            "authors": "H. Brendan McMahan et al."
        }
    
    def validate_hyperparameters(self) -> bool:
        """Validate algorithm hyperparameters."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.local_epochs <= 0:
            raise ValueError("Local epochs must be positive")
        
        return True


def create_fedavg_experiment(
    model_template: nn.Module,
    train_datasets: List[torch.utils.data.Dataset],
    **kwargs
) -> Tuple[List[Client], FedAvgAggregator]:
    """
    Create a complete FedAvg experiment setup.
    
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
    batch_size = kwargs.get('batch_size', 32)
    device = kwargs.get('device', 'cpu')
    weighted_aggregation = kwargs.get('weighted_aggregation', True)
    
    # Create algorithm instance
    algorithm = FedAvgAlgorithm(
        model_template=model_template,
        learning_rate=learning_rate,
        local_epochs=local_epochs,
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
    
    logger.info(f"Created FedAvg experiment with {len(clients)} clients")
    
    return clients, aggregator