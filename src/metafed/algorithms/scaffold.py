"""
SCAFFOLD (Stochastic Controlled Averaging for Federated Learning) Algorithm Implementation.

Paper: SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
Authors: Sai Praneeth Karimireddy et al.
"""

from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn
from ..core.client import SCAFFOLDClient
from ..core.aggregation import SCAFFOLDAggregator
import logging

logger = logging.getLogger(__name__)


class SCAFFOLDAlgorithm:
    """
    SCAFFOLD algorithm implementation with control variates.
    
    SCAFFOLD uses control variates to reduce client drift and improve
    convergence in heterogeneous federated learning settings.
    """
    
    def __init__(
        self,
        model_template: nn.Module,
        learning_rate: float = 0.01,
        local_epochs: int = 5,
        device: str = "cpu"
    ):
        """
        Initialize SCAFFOLD algorithm.
        
        Args:
            model_template: Neural network model template
            learning_rate: Client learning rate
            local_epochs: Number of local training epochs
            device: Device for computation
        """
        self.model_template = model_template
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.device = device
        
        # Initialize aggregator
        self.aggregator = SCAFFOLDAggregator()
        
        logger.info(f"Initialized SCAFFOLD algorithm with lr={learning_rate}, "
                   f"local_epochs={local_epochs}")
    
    def create_client(
        self,
        client_id: int,
        train_loader: torch.utils.data.DataLoader
    ) -> SCAFFOLDClient:
        """
        Create a SCAFFOLD client.
        
        Args:
            client_id: Unique client identifier
            train_loader: Training data loader for this client
            
        Returns:
            Configured SCAFFOLD client
        """
        return SCAFFOLDClient(
            client_id=client_id,
            train_loader=train_loader,
            model_template=self.model_template,
            lr=self.learning_rate,
            device=self.device,
            local_epochs=self.local_epochs
        )
    
    def get_aggregator(self) -> SCAFFOLDAggregator:
        """Get the SCAFFOLD aggregator."""
        return self.aggregator
    
    def update_server_control(self, clients: List[SCAFFOLDClient]) -> None:
        """
        Update server control variates.
        
        Args:
            clients: List of SCAFFOLD clients
        """
        server_control = self.aggregator.get_server_control()
        
        for client in clients:
            client.update_controls(server_control)
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information and hyperparameters."""
        return {
            "name": "SCAFFOLD",
            "learning_rate": self.learning_rate,
            "local_epochs": self.local_epochs,
            "paper": "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning",
            "year": 2020,
            "authors": "Sai Praneeth Karimireddy et al.",
            "key_features": [
                "Control variates to reduce client drift",
                "Improved convergence in non-IID settings",
                "Variance reduction technique",
                "Better theoretical guarantees"
            ],
            "advantages": [
                "Faster convergence than FedAvg",
                "Better handling of data heterogeneity",
                "Reduced communication rounds needed"
            ]
        }
    
    def validate_hyperparameters(self) -> bool:
        """Validate algorithm hyperparameters."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.local_epochs <= 0:
            raise ValueError("Local epochs must be positive")
        
        return True
    
    def get_control_variate_stats(self) -> Dict[str, Any]:
        """Get statistics about control variates."""
        server_control = self.aggregator.get_server_control()
        
        if not server_control:
            return {"status": "not_initialized"}
        
        stats = {}
        for name, control in server_control.items():
            stats[name] = {
                "mean": control.mean().item(),
                "std": control.std().item(),
                "norm": control.norm().item(),
                "shape": list(control.shape)
            }
        
        return stats


def create_scaffold_experiment(
    model_template: nn.Module,
    train_datasets: List[torch.utils.data.Dataset],
    **kwargs
) -> Tuple[List[SCAFFOLDClient], SCAFFOLDAggregator]:
    """
    Create a complete SCAFFOLD experiment setup.
    
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
    
    # Create algorithm instance
    algorithm = SCAFFOLDAlgorithm(
        model_template=model_template,
        learning_rate=learning_rate,
        local_epochs=local_epochs,
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
    
    logger.info(f"Created SCAFFOLD experiment with {len(clients)} clients")
    
    return clients, aggregator


def compare_scaffold_vs_fedavg(
    model_template: nn.Module,
    train_datasets: List[torch.utils.data.Dataset],
    test_loader: torch.utils.data.DataLoader,
    num_rounds: int = 50,
    **kwargs
) -> Dict[str, Any]:
    """
    Compare SCAFFOLD vs FedAvg performance.
    
    Args:
        model_template: Neural network model
        train_datasets: Training datasets for clients
        test_loader: Test data loader
        num_rounds: Number of rounds to run
        **kwargs: Additional parameters
        
    Returns:
        Comparison results
    """
    from .fedavg import create_fedavg_experiment
    
    logger.info("Running SCAFFOLD vs FedAvg comparison")
    
    # Create experiments
    scaffold_clients, scaffold_aggregator = create_scaffold_experiment(
        model_template, train_datasets, **kwargs
    )
    
    fedavg_clients, fedavg_aggregator = create_fedavg_experiment(
        model_template, train_datasets, **kwargs
    )
    
    # This is a simplified comparison framework
    # In practice, you would run full federated learning experiments
    
    results = {
        "scaffold": {
            "num_clients": len(scaffold_clients),
            "algorithm": "SCAFFOLD",
            "features": ["control_variates", "variance_reduction"]
        },
        "fedavg": {
            "num_clients": len(fedavg_clients),
            "algorithm": "FedAvg",
            "features": ["simple_averaging"]
        },
        "comparison_metrics": [
            "convergence_speed",
            "final_accuracy",
            "communication_rounds",
            "robustness_to_heterogeneity"
        ]
    }
    
    logger.info("Comparison setup completed - run full experiments for actual results")
    
    return results


class SCAFFOLDAnalyzer:
    """Analyzer for SCAFFOLD algorithm performance and behavior."""
    
    def __init__(self):
        self.control_history = []
        self.gradient_history = []
    
    def analyze_control_variates(self, aggregator: SCAFFOLDAggregator) -> Dict[str, Any]:
        """Analyze control variate evolution."""
        server_control = aggregator.get_server_control()
        
        analysis = {
            "num_parameters": sum(param.numel() for param in server_control.values()),
            "control_norms": {},
            "parameter_stats": {}
        }
        
        for name, control in server_control.items():
            analysis["control_norms"][name] = control.norm().item()
            analysis["parameter_stats"][name] = {
                "min": control.min().item(),
                "max": control.max().item(),
                "mean": control.mean().item(),
                "std": control.std().item()
            }
        
        return analysis
    
    def diagnose_convergence(self, loss_history: List[float]) -> Dict[str, Any]:
        """Diagnose convergence issues in SCAFFOLD."""
        if len(loss_history) < 5:
            return {"status": "insufficient_data"}
        
        # Simple convergence analysis
        recent_losses = loss_history[-5:]
        trend = "decreasing" if recent_losses[-1] < recent_losses[0] else "increasing"
        
        convergence_rate = abs(recent_losses[-1] - recent_losses[0]) / len(recent_losses)
        
        return {
            "trend": trend,
            "convergence_rate": convergence_rate,
            "is_converging": trend == "decreasing" and convergence_rate > 1e-6,
            "recommendations": self._get_convergence_recommendations(trend, convergence_rate)
        }
    
    def _get_convergence_recommendations(self, trend: str, rate: float) -> List[str]:
        """Get recommendations for improving convergence."""
        recommendations = []
        
        if trend == "increasing":
            recommendations.append("Consider reducing learning rate")
            recommendations.append("Check for gradient explosion")
        
        if rate < 1e-6:
            recommendations.append("Learning rate might be too small")
            recommendations.append("Consider adjusting local epochs")
        
        return recommendations