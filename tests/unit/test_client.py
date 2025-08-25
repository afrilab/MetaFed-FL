"""
Unit tests for federated learning clients.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from metafed.core.client import Client, FedProxClient, SCAFFOLDClient


class TestClient:
    """Test cases for the base Client class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X = torch.randn(50, 10)
        y = torch.randint(0, 2, (50,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=8, shuffle=True)
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 5)
                self.fc2 = nn.Linear(5, 2)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)
        
        return SimpleModel()
    
    def test_client_initialization(self, sample_data, simple_model):
        """Test client initialization."""
        client = Client(
            client_id=0,
            train_loader=sample_data,
            model_template=simple_model,
            lr=0.01,
            device="cpu",
            local_epochs=1
        )
        
        assert client.id == 0
        assert client.lr == 0.01
        assert client.local_epochs == 1
        assert client.device == "cpu"
        assert isinstance(client.model, nn.Module)
    
    def test_client_model_update(self, sample_data, simple_model):
        """Test client model parameter update."""
        client = Client(
            client_id=0,
            train_loader=sample_data,
            model_template=simple_model,
            lr=0.01,
            device="cpu"
        )
        
        # Get initial parameters
        initial_params = client.get_model_params()
        
        # Create new parameters
        new_params = {}
        for name, param in initial_params.items():
            new_params[name] = param + 0.1  # Add small value
        
        # Update model
        client.update_model(new_params)
        updated_params = client.get_model_params()
        
        # Check that parameters were updated
        for name in initial_params.keys():
            assert not torch.equal(initial_params[name], updated_params[name])
    
    def test_client_training(self, sample_data, simple_model):
        """Test client training process."""
        client = Client(
            client_id=0,
            train_loader=sample_data,
            model_template=simple_model,
            lr=0.01,
            device="cpu",
            local_epochs=1
        )
        
        # Train client
        model_params, num_samples, avg_loss = client.train()
        
        # Check return values
        assert isinstance(model_params, dict)
        assert isinstance(num_samples, int)
        assert isinstance(avg_loss, float)
        assert num_samples > 0
        assert avg_loss >= 0


class TestFedProxClient:
    """Test cases for FedProx client."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=8, shuffle=True)
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 2)
            
            def forward(self, x):
                return self.fc(x)
        
        return SimpleModel()
    
    def test_fedprox_initialization(self, sample_data, simple_model):
        """Test FedProx client initialization."""
        client = FedProxClient(
            client_id=1,
            train_loader=sample_data,
            model_template=simple_model,
            lr=0.01,
            device="cpu",
            mu=0.01
        )
        
        assert client.id == 1
        assert client.mu == 0.01
        assert client.global_params is None
    
    def test_fedprox_global_params_storage(self, sample_data, simple_model):
        """Test global parameter storage in FedProx."""
        client = FedProxClient(
            client_id=1,
            train_loader=sample_data,
            model_template=simple_model,
            lr=0.01,
            device="cpu",
            mu=0.01
        )
        
        # Update with global parameters
        global_params = client.get_model_params()
        client.update_model(global_params)
        
        # Check that global parameters are stored
        assert client.global_params is not None
        assert len(client.global_params) == len(global_params)
    
    def test_fedprox_training_with_proximal_term(self, sample_data, simple_model):
        """Test FedProx training with proximal term."""
        client = FedProxClient(
            client_id=1,
            train_loader=sample_data,
            model_template=simple_model,
            lr=0.01,
            device="cpu",
            mu=0.01,
            local_epochs=1
        )
        
        # Set global parameters first
        global_params = client.get_model_params()
        client.update_model(global_params)
        
        # Train with proximal term
        model_params, num_samples, avg_loss = client.train()
        
        # Check return values
        assert isinstance(model_params, dict)
        assert isinstance(num_samples, int)
        assert isinstance(avg_loss, float)


class TestSCAFFOLDClient:
    """Test cases for SCAFFOLD client."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X = torch.randn(24, 10)
        y = torch.randint(0, 2, (24,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=8, shuffle=True)
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 2)
            
            def forward(self, x):
                return self.fc(x)
        
        return SimpleModel()
    
    def test_scaffold_initialization(self, sample_data, simple_model):
        """Test SCAFFOLD client initialization."""
        client = SCAFFOLDClient(
            client_id=2,
            train_loader=sample_data,
            model_template=simple_model,
            lr=0.01,
            device="cpu"
        )
        
        assert client.id == 2
        assert client.client_control is not None
        assert client.server_control is not None
        assert len(client.client_control) > 0
    
    def test_scaffold_control_variates_update(self, sample_data, simple_model):
        """Test control variates update in SCAFFOLD."""
        client = SCAFFOLDClient(
            client_id=2,
            train_loader=sample_data,
            model_template=simple_model,
            lr=0.01,
            device="cpu"
        )
        
        # Create mock server control
        server_control = {}
        for name, param in client.model.named_parameters():
            server_control[name] = torch.zeros_like(param)
        
        # Update controls
        client.update_controls(server_control)
        
        # Check that server control is updated
        for name in server_control.keys():
            assert name in client.server_control
    
    def test_scaffold_training_returns_control_delta(self, sample_data, simple_model):
        """Test that SCAFFOLD training returns control delta."""
        client = SCAFFOLDClient(
            client_id=2,
            train_loader=sample_data,
            model_template=simple_model,
            lr=0.01,
            device="cpu",
            local_epochs=1
        )
        
        # Train client
        result = client.train()
        
        # SCAFFOLD should return 4 values including control delta
        assert len(result) == 4
        model_params, control_delta, num_samples, avg_loss = result
        
        assert isinstance(model_params, dict)
        assert isinstance(control_delta, dict)
        assert isinstance(num_samples, int)
        assert isinstance(avg_loss, float)


@pytest.mark.parametrize("client_class,extra_kwargs", [
    (Client, {}),
    (FedProxClient, {"mu": 0.01}),
    (SCAFFOLDClient, {})
])
def test_all_client_types_basic_functionality(client_class, extra_kwargs):
    """Test basic functionality across all client types."""
    # Create simple data and model
    X = torch.randn(16, 5)
    y = torch.randint(0, 2, (16,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=8)
    
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(5, 2)
        
        def forward(self, x):
            return self.fc(x)
    
    model = TinyModel()
    
    # Create client
    client = client_class(
        client_id=0,
        train_loader=dataloader,
        model_template=model,
        lr=0.01,
        device="cpu",
        local_epochs=1,
        **extra_kwargs
    )
    
    # Test basic operations
    params = client.get_model_params()
    assert isinstance(params, dict)
    assert len(params) > 0
    
    # Test training
    result = client.train()
    assert len(result) >= 3  # At least params, num_samples, loss


if __name__ == "__main__":
    pytest.main([__file__])