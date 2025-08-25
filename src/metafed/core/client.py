"""
Federated Learning Client Implementations.

This module contains different client implementations for various
federated learning algorithms including FedAvg, FedProx, and SCAFFOLD.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple
import copy
import logging

logger = logging.getLogger(__name__)


class Client:
    """
    Base federated learning client implementation.
    
    This class provides the fundamental structure for federated learning clients
    including training, model updates, and parameter management.
    """
    
    def __init__(
        self,
        client_id: int,
        train_loader: torch.utils.data.DataLoader,
        model_template: nn.Module,
        lr: float = 0.01,
        device: str = "cpu",
        local_epochs: int = 5
    ):
        """
        Initialize a federated learning client.
        
        Args:
            client_id: Unique identifier for this client
            train_loader: DataLoader for training data
            model_template: Neural network model template
            lr: Learning rate for local training
            device: Device to run computations on ('cpu' or 'cuda')
            local_epochs: Number of local training epochs
        """
        self.id = client_id
        self.train_loader = train_loader
        self.model = copy.deepcopy(model_template)
        self.lr = lr
        self.device = device
        self.local_epochs = local_epochs
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Initialized client {self.id} on device {self.device}")
    
    def update_model(self, global_params: Dict[str, torch.Tensor]) -> None:
        """
        Update local model with global parameters.
        
        Args:
            global_params: Dictionary of global model parameters
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in global_params:
                    param.copy_(global_params[name])
    
    def get_model_params(self) -> Dict[str, torch.Tensor]:
        """
        Get current model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return {name: param.clone() for name, param in self.model.named_parameters()}
    
    def train(self) -> Tuple[Dict[str, torch.Tensor], int, float]:
        """
        Perform local training.
        
        Returns:
            Tuple of (model_params, num_samples, avg_loss)
        """
        self.model.train()
        total_loss = 0.0
        num_samples = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * data.size(0)
                epoch_samples += data.size(0)
            
            total_loss += epoch_loss
            num_samples += epoch_samples
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        logger.info(f"Client {self.id} training complete. Loss: {avg_loss:.4f}")
        
        return self.get_model_params(), num_samples, avg_loss


class FedProxClient(Client):
    """
    FedProx client implementation with proximal term.
    
    FedProx adds a proximal term to the loss function to handle
    system heterogeneity and partial work.
    """
    
    def __init__(
        self,
        client_id: int,
        train_loader: torch.utils.data.DataLoader,
        model_template: nn.Module,
        lr: float = 0.01,
        device: str = "cpu",
        local_epochs: int = 5,
        mu: float = 0.01
    ):
        """
        Initialize FedProx client.
        
        Args:
            mu: Proximal term coefficient
        """
        super().__init__(client_id, train_loader, model_template, lr, device, local_epochs)
        self.mu = mu
        self.global_params = None
        
        logger.info(f"Initialized FedProx client {self.id} with mu={self.mu}")
    
    def update_model(self, global_params: Dict[str, torch.Tensor]) -> None:
        """Update model and store global parameters for proximal term."""
        super().update_model(global_params)
        # Store global parameters for proximal term
        self.global_params = {name: param.clone() for name, param in global_params.items()}
    
    def train(self) -> Tuple[Dict[str, torch.Tensor], int, float]:
        """
        Perform local training with proximal term.
        
        Returns:
            Tuple of (model_params, num_samples, avg_loss)
        """
        if self.global_params is None:
            return super().train()
        
        self.model.train()
        total_loss = 0.0
        num_samples = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Add proximal term
                proximal_term = 0.0
                for name, param in self.model.named_parameters():
                    if name in self.global_params:
                        proximal_term += torch.norm(param - self.global_params[name]) ** 2
                
                loss += (self.mu / 2) * proximal_term
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * data.size(0)
                epoch_samples += data.size(0)
            
            total_loss += epoch_loss
            num_samples += epoch_samples
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        logger.info(f"FedProx Client {self.id} training complete. Loss: {avg_loss:.4f}")
        
        return self.get_model_params(), num_samples, avg_loss


class SCAFFOLDClient(Client):
    """
    SCAFFOLD client implementation with control variates.
    
    SCAFFOLD uses control variates to reduce client drift
    and improve convergence in heterogeneous settings.
    """
    
    def __init__(
        self,
        client_id: int,
        train_loader: torch.utils.data.DataLoader,
        model_template: nn.Module,
        lr: float = 0.01,
        device: str = "cpu",
        local_epochs: int = 5
    ):
        """Initialize SCAFFOLD client."""
        super().__init__(client_id, train_loader, model_template, lr, device, local_epochs)
        
        # Initialize control variates
        self.client_control = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        self.server_control = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        
        logger.info(f"Initialized SCAFFOLD client {self.id}")
    
    def update_controls(self, server_control: Dict[str, torch.Tensor]) -> None:
        """
        Update server control variates.
        
        Args:
            server_control: Server control variates
        """
        self.server_control = {name: param.clone() for name, param in server_control.items()}
    
    def train(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, float]:
        """
        Perform local training with control variates.
        
        Returns:
            Tuple of (model_params, client_control_delta, num_samples, avg_loss)
        """
        # Store initial parameters
        initial_params = self.get_model_params()
        
        self.model.train()
        total_loss = 0.0
        num_samples = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                # Apply SCAFFOLD correction
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            correction = self.server_control[name] - self.client_control[name]
                            param.grad += correction
                
                self.optimizer.step()
                
                epoch_loss += loss.item() * data.size(0)
                epoch_samples += data.size(0)
            
            total_loss += epoch_loss
            num_samples += epoch_samples
        
        # Update client control variates
        final_params = self.get_model_params()
        new_client_control = {}
        
        for name in self.client_control.keys():
            option_1 = self.server_control[name] - self.client_control[name]
            option_2 = (initial_params[name] - final_params[name]) / (self.local_epochs * self.lr)
            new_client_control[name] = option_1 + option_2
        
        # Calculate control variate delta
        control_delta = {}
        for name in self.client_control.keys():
            control_delta[name] = new_client_control[name] - self.client_control[name]
        
        # Update stored control variates
        self.client_control = new_client_control
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        logger.info(f"SCAFFOLD Client {self.id} training complete. Loss: {avg_loss:.4f}")
        
        return final_params, control_delta, num_samples, avg_loss