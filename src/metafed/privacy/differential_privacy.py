"""
Differential Privacy implementation for Federated Learning.

This module provides differential privacy mechanisms for preserving client privacy
in federated learning through noise addition and privacy accounting.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
import math

logger = logging.getLogger(__name__)


class DifferentialPrivacy:
    """
    Differential Privacy mechanism for federated learning.
    
    Implements various DP mechanisms including Gaussian mechanism,
    Laplace mechanism, and advanced composition techniques.
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
        mechanism: str = "gaussian",
        clip_norm: float = 1.0
    ):
        """
        Initialize differential privacy mechanism.
        
        Args:
            epsilon: Privacy parameter (smaller = more private)
            delta: Privacy parameter for (ε,δ)-DP
            sensitivity: L2 sensitivity of the function
            mechanism: DP mechanism ("gaussian", "laplace")
            clip_norm: Gradient clipping norm
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.mechanism = mechanism
        self.clip_norm = clip_norm
        
        # Privacy accounting
        self.privacy_spent = 0.0
        self.composition_history = []
        
        logger.info(f"Initialized DP with ε={epsilon}, δ={delta}, mechanism={mechanism}")
    
    def add_noise(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Add differential privacy noise to model parameters.
        
        Args:
            params: Model parameters
            
        Returns:
            Parameters with DP noise added
        """
        if self.mechanism == "gaussian":
            return self._add_gaussian_noise(params)
        elif self.mechanism == "laplace":
            return self._add_laplace_noise(params)
        else:
            raise ValueError(f"Unknown DP mechanism: {self.mechanism}")
    
    def _add_gaussian_noise(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add Gaussian noise for (ε,δ)-differential privacy."""
        # Calculate noise scale for Gaussian mechanism
        if self.delta <= 0:
            raise ValueError("Delta must be positive for Gaussian mechanism")
        
        # Standard formula: σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        noise_scale = self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        
        noisy_params = {}
        total_noise_norm = 0.0
        
        for name, param in params.items():
            # Generate Gaussian noise
            noise = torch.normal(0, noise_scale, size=param.shape, device=param.device)
            noisy_params[name] = param + noise
            
            # Track noise statistics
            total_noise_norm += noise.norm().item() ** 2
        
        total_noise_norm = math.sqrt(total_noise_norm)
        
        # Update privacy accounting
        self.privacy_spent += self.epsilon
        self.composition_history.append({
            "epsilon": self.epsilon,
            "delta": self.delta,
            "noise_scale": noise_scale,
            "total_noise_norm": total_noise_norm
        })
        
        logger.debug(f"Added Gaussian noise with scale {noise_scale:.6f}")
        
        return noisy_params
    
    def _add_laplace_noise(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add Laplace noise for ε-differential privacy."""
        # Laplace mechanism: scale = sensitivity / ε
        noise_scale = self.sensitivity / self.epsilon
        
        noisy_params = {}
        
        for name, param in params.items():
            # Generate Laplace noise
            noise = torch.distributions.Laplace(0, noise_scale).sample(param.shape)
            noise = noise.to(param.device)
            noisy_params[name] = param + noise
        
        # Update privacy accounting
        self.privacy_spent += self.epsilon
        self.composition_history.append({
            "epsilon": self.epsilon,
            "delta": 0.0,  # Pure DP
            "noise_scale": noise_scale
        })
        
        logger.debug(f"Added Laplace noise with scale {noise_scale:.6f}")
        
        return noisy_params
    
    def clip_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Clip gradients to bound sensitivity.
        
        Args:
            gradients: Gradient tensors
            
        Returns:
            Clipped gradients
        """
        # Calculate total gradient norm
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += grad.norm().item() ** 2
        total_norm = math.sqrt(total_norm)
        
        # Clip if necessary
        if total_norm > self.clip_norm:
            clip_factor = self.clip_norm / total_norm
            clipped_gradients = {}
            for name, grad in gradients.items():
                clipped_gradients[name] = grad * clip_factor
            
            logger.debug(f"Clipped gradients: norm {total_norm:.4f} -> {self.clip_norm:.4f}")
            return clipped_gradients
        
        return gradients
    
    def get_privacy_spent(self) -> float:
        """Get total privacy budget spent."""
        return self.privacy_spent
    
    def get_remaining_budget(self, total_budget: float) -> float:
        """Get remaining privacy budget."""
        return max(0.0, total_budget - self.privacy_spent)
    
    def reset_privacy_accounting(self) -> None:
        """Reset privacy accounting."""
        self.privacy_spent = 0.0
        self.composition_history = []
        logger.info("Privacy accounting reset")
    
    def compute_rdp_epsilon(self, orders: List[float], num_steps: int) -> float:
        """
        Compute privacy cost using Rényi Differential Privacy.
        
        Args:
            orders: List of Rényi orders
            num_steps: Number of steps/rounds
            
        Returns:
            Epsilon for given delta using RDP
        """
        # This is a simplified RDP computation
        # In practice, you would use libraries like autodp or opacus
        
        if self.mechanism != "gaussian":
            raise ValueError("RDP computation only supported for Gaussian mechanism")
        
        noise_scale = self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        
        # Simplified RDP computation for Gaussian mechanism
        rdp_values = []
        for order in orders:
            if order == 1:
                continue  # RDP is not defined for order 1
            
            # For Gaussian mechanism: RDP(α) = α / (2 * σ²)
            rdp = order / (2 * noise_scale ** 2)
            rdp_values.append(rdp)
        
        # Convert RDP to (ε,δ)-DP
        if rdp_values:
            # Simplified conversion (use proper RDP libraries in practice)
            min_rdp = min(rdp_values)
            eps = min_rdp * num_steps + math.log(1 / self.delta) / (min(orders[1:]) - 1)
            return eps
        
        return float('inf')


class PrivacyAccountant:
    """
    Privacy accountant for tracking cumulative privacy loss.
    
    Supports various composition theorems and privacy amplification techniques.
    """
    
    def __init__(self, total_epsilon: float, total_delta: float):
        """
        Initialize privacy accountant.
        
        Args:
            total_epsilon: Total privacy budget
            total_delta: Total delta budget
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.transactions = []
        
        logger.info(f"Privacy accountant initialized with budget (ε={total_epsilon}, δ={total_delta})")
    
    def spend_privacy(self, epsilon: float, delta: float, description: str = "") -> bool:
        """
        Spend privacy budget and check if within limits.
        
        Args:
            epsilon: Epsilon to spend
            delta: Delta to spend
            description: Description of the transaction
            
        Returns:
            True if transaction successful, False if budget exceeded
        """
        new_epsilon = self.spent_epsilon + epsilon
        new_delta = self.spent_delta + delta
        
        if new_epsilon > self.total_epsilon or new_delta > self.total_delta:
            logger.warning(f"Privacy budget exceeded: (ε={new_epsilon:.4f}/{self.total_epsilon}, "
                          f"δ={new_delta:.8f}/{self.total_delta})")
            return False
        
        self.spent_epsilon = new_epsilon
        self.spent_delta = new_delta
        
        self.transactions.append({
            "epsilon": epsilon,
            "delta": delta,
            "description": description,
            "cumulative_epsilon": self.spent_epsilon,
            "cumulative_delta": self.spent_delta
        })
        
        logger.info(f"Privacy spent: ε={epsilon:.4f}, δ={delta:.8f}. "
                   f"Remaining: ε={self.get_remaining_epsilon():.4f}, δ={self.get_remaining_delta():.8f}")
        
        return True
    
    def get_remaining_epsilon(self) -> float:
        """Get remaining epsilon budget."""
        return max(0.0, self.total_epsilon - self.spent_epsilon)
    
    def get_remaining_delta(self) -> float:
        """Get remaining delta budget."""
        return max(0.0, self.total_delta - self.spent_delta)
    
    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get comprehensive privacy summary."""
        return {
            "total_budget": {"epsilon": self.total_epsilon, "delta": self.total_delta},
            "spent_budget": {"epsilon": self.spent_epsilon, "delta": self.spent_delta},
            "remaining_budget": {
                "epsilon": self.get_remaining_epsilon(),
                "delta": self.get_remaining_delta()
            },
            "num_transactions": len(self.transactions),
            "transactions": self.transactions
        }


class FederatedDifferentialPrivacy:
    """
    Federated learning specific differential privacy implementation.
    
    Handles client-level DP, server-side aggregation with DP,
    and privacy amplification through subsampling.
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        num_clients: int = 100,
        sampling_rate: float = 0.1,
        clip_norm: float = 1.0
    ):
        """
        Initialize federated DP.
        
        Args:
            epsilon: Privacy parameter
            delta: Privacy parameter
            num_clients: Total number of clients
            sampling_rate: Client sampling rate
            clip_norm: Gradient clipping norm
        """
        self.epsilon = epsilon
        self.delta = delta
        self.num_clients = num_clients
        self.sampling_rate = sampling_rate
        self.clip_norm = clip_norm
        
        # Calculate noise scale with privacy amplification
        self.effective_epsilon = self._compute_amplified_epsilon()
        
        # Initialize DP mechanism
        self.dp_mechanism = DifferentialPrivacy(
            epsilon=self.effective_epsilon,
            delta=delta,
            sensitivity=clip_norm,  # After clipping, sensitivity is clip_norm
            clip_norm=clip_norm
        )
        
        # Privacy accountant
        self.accountant = PrivacyAccountant(total_epsilon=epsilon, total_delta=delta)
        
        logger.info(f"Federated DP initialized: original ε={epsilon}, "
                   f"amplified ε={self.effective_epsilon:.4f}")
    
    def _compute_amplified_epsilon(self) -> float:
        """
        Compute privacy amplification due to subsampling.
        
        Returns:
            Amplified epsilon value
        """
        # Privacy amplification by subsampling
        # For uniform sampling: ε' ≈ ε * sampling_rate (simplified)
        # More precise bounds exist in literature
        
        amplification_factor = self.sampling_rate
        amplified_epsilon = self.epsilon * amplification_factor
        
        logger.debug(f"Privacy amplification: {self.epsilon} -> {amplified_epsilon} "
                    f"(factor: {amplification_factor})")
        
        return amplified_epsilon
    
    def privatize_aggregation(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        round_num: int
    ) -> Dict[str, torch.Tensor]:
        """
        Apply differential privacy to federated aggregation.
        
        Args:
            client_updates: List of client model updates
            round_num: Current round number
            
        Returns:
            Privatized aggregated parameters
        """
        # First, perform standard aggregation
        aggregated_params = self._aggregate_updates(client_updates)
        
        # Clip the aggregated gradients
        clipped_params = self.dp_mechanism.clip_gradients(aggregated_params)
        
        # Add noise
        private_params = self.dp_mechanism.add_noise(clipped_params)
        
        # Update privacy accounting
        success = self.accountant.spend_privacy(
            epsilon=self.effective_epsilon,
            delta=self.delta,
            description=f"Round {round_num} aggregation"
        )
        
        if not success:
            logger.error(f"Privacy budget exhausted at round {round_num}")
            raise ValueError("Privacy budget exhausted")
        
        return private_params
    
    def _aggregate_updates(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Simple averaging of client updates."""
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        # Initialize aggregated parameters
        aggregated = {}
        first_update = client_updates[0]
        
        for name, param in first_update.items():
            aggregated[name] = torch.zeros_like(param)
        
        # Average all updates
        for update in client_updates:
            for name, param in update.items():
                aggregated[name] += param / len(client_updates)
        
        return aggregated
    
    def check_privacy_budget(self) -> Dict[str, Any]:
        """Check remaining privacy budget."""
        return self.accountant.get_privacy_summary()
    
    def estimate_max_rounds(self) -> int:
        """Estimate maximum number of rounds with current privacy budget."""
        remaining_epsilon = self.accountant.get_remaining_epsilon()
        if self.effective_epsilon <= 0:
            return 0
        
        max_rounds = int(remaining_epsilon / self.effective_epsilon)
        return max(0, max_rounds)


def create_dp_federated_experiment(
    epsilon: float = 1.0,
    delta: float = 1e-5,
    num_clients: int = 100,
    clients_per_round: int = 10,
    clip_norm: float = 1.0
) -> FederatedDifferentialPrivacy:
    """
    Create a differential privacy setup for federated learning.
    
    Args:
        epsilon: Privacy parameter
        delta: Privacy parameter  
        num_clients: Total number of clients
        clients_per_round: Clients selected per round
        clip_norm: Gradient clipping norm
        
    Returns:
        Configured federated DP instance
    """
    sampling_rate = clients_per_round / num_clients
    
    dp_fed = FederatedDifferentialPrivacy(
        epsilon=epsilon,
        delta=delta,
        num_clients=num_clients,
        sampling_rate=sampling_rate,
        clip_norm=clip_norm
    )
    
    logger.info(f"Created DP federated experiment with (ε={epsilon}, δ={delta})")
    logger.info(f"Estimated max rounds: {dp_fed.estimate_max_rounds()}")
    
    return dp_fed