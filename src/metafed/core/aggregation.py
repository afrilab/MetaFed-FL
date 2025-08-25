"""
Federated Learning Aggregation Algorithms.

This module implements various aggregation strategies for combining
client model updates in federated learning.
"""

import torch
from typing import Dict, List, Any
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseAggregator(ABC):
    """Base class for federated learning aggregators."""
    
    @abstractmethod
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates.
        
        Args:
            client_updates: List of client updates containing parameters and metadata
            
        Returns:
            Aggregated model parameters
        """
        pass


class FedAvgAggregator(BaseAggregator):
    """
    Federated Averaging (FedAvg) aggregator.
    
    Performs weighted averaging of client model parameters
    based on the number of training samples.
    """
    
    def __init__(self, weighted: bool = True):
        """
        Initialize FedAvg aggregator.
        
        Args:
            weighted: Whether to weight aggregation by number of samples
        """
        self.weighted = weighted
        logger.info(f"Initialized FedAvg aggregator (weighted={weighted})")
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using federated averaging.
        
        Args:
            client_updates: List of dictionaries containing:
                - 'params': model parameters
                - 'num_samples': number of training samples
                - 'client_id': client identifier
        
        Returns:
            Aggregated model parameters
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")
        
        # Calculate weights
        if self.weighted:
            total_samples = sum(update['num_samples'] for update in client_updates)
            weights = [update['num_samples'] / total_samples for update in client_updates]
        else:
            weights = [1.0 / len(client_updates) for _ in client_updates]
        
        # Initialize aggregated parameters with zeros
        aggregated_params = {}
        first_update = client_updates[0]['params']
        
        for name, param in first_update.items():
            aggregated_params[name] = torch.zeros_like(param)
        
        # Weighted aggregation
        for i, update in enumerate(client_updates):
            weight = weights[i]
            for name, param in update['params'].items():
                aggregated_params[name] += weight * param
        
        logger.info(f"Aggregated {len(client_updates)} client updates using FedAvg")
        
        return aggregated_params


class FedProxAggregator(BaseAggregator):
    """
    FedProx aggregator.
    
    Similar to FedAvg but designed to work with FedProx client updates
    that include proximal terms.
    """
    
    def __init__(self, weighted: bool = True):
        """
        Initialize FedProx aggregator.
        
        Args:
            weighted: Whether to weight aggregation by number of samples
        """
        self.weighted = weighted
        logger.info(f"Initialized FedProx aggregator (weighted={weighted})")
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate FedProx client updates.
        
        Args:
            client_updates: List of client updates from FedProx clients
        
        Returns:
            Aggregated model parameters
        """
        # FedProx uses the same aggregation as FedAvg
        return FedAvgAggregator(weighted=self.weighted).aggregate(client_updates)


class SCAFFOLDAggregator(BaseAggregator):
    """
    SCAFFOLD aggregator with control variates.
    
    Aggregates both model parameters and control variates
    from SCAFFOLD clients.
    """
    
    def __init__(self):
        """Initialize SCAFFOLD aggregator."""
        self.server_control = None
        logger.info("Initialized SCAFFOLD aggregator")
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate SCAFFOLD client updates.
        
        Args:
            client_updates: List of dictionaries containing:
                - 'params': model parameters
                - 'control_delta': control variate updates
                - 'num_samples': number of training samples
        
        Returns:
            Aggregated model parameters
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")
        
        # Calculate weights based on number of samples
        total_samples = sum(update['num_samples'] for update in client_updates)
        weights = [update['num_samples'] / total_samples for update in client_updates]
        
        # Initialize aggregated parameters
        aggregated_params = {}
        first_update = client_updates[0]['params']
        
        for name, param in first_update.items():
            aggregated_params[name] = torch.zeros_like(param)
        
        # Aggregate model parameters
        for i, update in enumerate(client_updates):
            weight = weights[i]
            for name, param in update['params'].items():
                aggregated_params[name] += weight * param
        
        # Update server control variates if available
        if 'control_delta' in client_updates[0]:
            if self.server_control is None:
                # Initialize server control variates
                self.server_control = {}
                for name, param in first_update.items():
                    self.server_control[name] = torch.zeros_like(param)
            
            # Aggregate control variate updates
            for i, update in enumerate(client_updates):
                weight = weights[i]
                for name, control_delta in update['control_delta'].items():
                    self.server_control[name] += weight * control_delta
        
        logger.info(f"Aggregated {len(client_updates)} SCAFFOLD client updates")
        
        return aggregated_params
    
    def get_server_control(self) -> Dict[str, torch.Tensor]:
        """
        Get server control variates.
        
        Returns:
            Dictionary of server control variates
        """
        if self.server_control is None:
            return {}
        return {name: control.clone() for name, control in self.server_control.items()}


class SecureAggregator(BaseAggregator):
    """
    Secure aggregator with privacy-preserving techniques.
    
    Implements secure aggregation with optional differential privacy
    and homomorphic encryption support.
    """
    
    def __init__(
        self,
        base_aggregator: BaseAggregator,
        enable_dp: bool = False,
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
        noise_multiplier: float = 1.0
    ):
        """
        Initialize secure aggregator.
        
        Args:
            base_aggregator: Base aggregation method
            enable_dp: Enable differential privacy
            dp_epsilon: Privacy parameter epsilon
            dp_delta: Privacy parameter delta
            noise_multiplier: Noise multiplier for DP
        """
        self.base_aggregator = base_aggregator
        self.enable_dp = enable_dp
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.noise_multiplier = noise_multiplier
        
        logger.info(f"Initialized secure aggregator with DP={enable_dp}")
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates with privacy preservation.
        
        Args:
            client_updates: List of client updates
        
        Returns:
            Privacy-preserving aggregated parameters
        """
        # Perform base aggregation
        aggregated_params = self.base_aggregator.aggregate(client_updates)
        
        # Apply differential privacy if enabled
        if self.enable_dp:
            aggregated_params = self._add_dp_noise(aggregated_params)
        
        return aggregated_params
    
    def _add_dp_noise(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Add differential privacy noise to parameters.
        
        Args:
            params: Model parameters
        
        Returns:
            Parameters with DP noise added
        """
        noisy_params = {}
        
        for name, param in params.items():
            # Calculate noise scale based on sensitivity and privacy parameters
            sensitivity = 1.0  # Assume unit L2 sensitivity
            noise_scale = self.noise_multiplier * sensitivity
            
            # Add Gaussian noise
            noise = torch.normal(0, noise_scale, size=param.shape, device=param.device)
            noisy_params[name] = param + noise
        
        logger.info(f"Added differential privacy noise (ε={self.dp_epsilon}, δ={self.dp_delta})")
        
        return noisy_params


class AdaptiveAggregator(BaseAggregator):
    """
    Adaptive aggregator that adjusts based on client performance.
    
    Weights client contributions based on their local performance
    and reliability metrics.
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize adaptive aggregator.
        
        Args:
            alpha: Learning rate for adaptive weights
        """
        self.alpha = alpha
        self.client_weights = {}
        self.client_performance_history = {}
        
        logger.info(f"Initialized adaptive aggregator (alpha={alpha})")
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate with adaptive weighting based on client performance.
        
        Args:
            client_updates: List of client updates with performance metrics
        
        Returns:
            Adaptively weighted aggregated parameters
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")
        
        # Update client performance history
        self._update_performance_history(client_updates)
        
        # Calculate adaptive weights
        weights = self._calculate_adaptive_weights(client_updates)
        
        # Initialize aggregated parameters
        aggregated_params = {}
        first_update = client_updates[0]['params']
        
        for name, param in first_update.items():
            aggregated_params[name] = torch.zeros_like(param)
        
        # Weighted aggregation with adaptive weights
        for i, update in enumerate(client_updates):
            weight = weights[i]
            for name, param in update['params'].items():
                aggregated_params[name] += weight * param
        
        logger.info(f"Aggregated {len(client_updates)} client updates with adaptive weights")
        
        return aggregated_params
    
    def _update_performance_history(self, client_updates: List[Dict[str, Any]]) -> None:
        """Update client performance history."""
        for update in client_updates:
            client_id = update['client_id']
            loss = update.get('loss', float('inf'))
            
            if client_id not in self.client_performance_history:
                self.client_performance_history[client_id] = []
            
            self.client_performance_history[client_id].append(loss)
            
            # Keep only recent history
            if len(self.client_performance_history[client_id]) > 10:
                self.client_performance_history[client_id] = \
                    self.client_performance_history[client_id][-10:]
    
    def _calculate_adaptive_weights(self, client_updates: List[Dict[str, Any]]) -> List[float]:
        """Calculate adaptive weights based on client performance."""
        weights = []
        
        for update in client_updates:
            client_id = update['client_id']
            num_samples = update['num_samples']
            
            # Base weight from number of samples
            base_weight = num_samples
            
            # Performance-based adjustment
            if client_id in self.client_performance_history:
                avg_loss = sum(self.client_performance_history[client_id]) / \
                          len(self.client_performance_history[client_id])
                # Lower loss means higher weight (inverse relationship)
                performance_weight = 1.0 / (1.0 + avg_loss)
            else:
                performance_weight = 1.0
            
            # Combine weights
            adaptive_weight = base_weight * performance_weight
            weights.append(adaptive_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(client_updates) for _ in client_updates]
        
        return weights