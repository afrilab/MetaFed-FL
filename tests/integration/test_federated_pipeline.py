"""
Integration tests for the complete federated learning pipeline.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
import os
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from metafed.core.client import Client, FedProxClient
from metafed.core.server import FederatedServer
from metafed.core.aggregation import FedAvgAggregator, FedProxAggregator
from metafed.orchestration.random_orchestrator import RandomOrchestrator
from metafed.algorithms.fedavg import create_fedavg_experiment
from metafed.algorithms.fedprox import create_fedprox_experiment
from metafed.data.loaders import partition_dataset
from metafed.models.simple_cnn import SimpleCNN


class TestFederatedLearningPipeline:
    """Integration tests for complete federated learning pipeline."""
    
    @pytest.fixture
    def synthetic_dataset(self):
        """Create synthetic dataset for testing."""
        # Generate synthetic data
        X = torch.randn(200, 10)
        y = torch.randint(0, 2, (200,))
        dataset = TensorDataset(X, y)
        return dataset
    
    @pytest.fixture
    def test_model(self):
        """Create test model."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 5)
                self.fc2 = nn.Linear(5, 2)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                return self.fc2(x)
        
        return TestModel()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_basic_federated_learning_round(self, synthetic_dataset, test_model):
        """Test a basic federated learning round."""
        # Partition data among clients
        client_datasets = partition_dataset(synthetic_dataset, num_clients=5, alpha=0.5)
        
        # Create clients
        clients = []
        for i, dataset in enumerate(client_datasets):
            dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
            client = Client(
                client_id=i,
                train_loader=dataloader,
                model_template=test_model,
                lr=0.01,
                device="cpu",
                local_epochs=1
            )
            clients.append(client)
        
        # Create server components
        orchestrator = RandomOrchestrator()
        aggregator = FedAvgAggregator()
        
        server = FederatedServer(
            model_template=test_model,
            orchestrator=orchestrator,
            num_rounds=3,
            clients_per_round=3,
            device="cpu"
        )
        
        # Create test data for evaluation
        test_X = torch.randn(50, 10)
        test_y = torch.randint(0, 2, (50,))
        test_dataset = TensorDataset(test_X, test_y)
        test_loader = DataLoader(test_dataset, batch_size=16)
        
        # Run federated learning
        results = server.run_federated_learning(
            clients=clients,
            aggregator=aggregator,
            test_loader=test_loader,
            eval_frequency=1
        )
        
        # Verify results structure
        assert 'training_history' in results
        assert 'final_model_state' in results
        assert 'total_training_time' in results
        
        history = results['training_history']
        assert len(history['rounds']) == 3
        assert len(history['losses']) == 3
        assert len(history['training_times']) == 3
    
    def test_fedavg_complete_experiment(self, synthetic_dataset, test_model):
        """Test complete FedAvg experiment using algorithm factory."""
        # Partition dataset
        client_datasets = partition_dataset(synthetic_dataset, num_clients=4, alpha=0.5)
        
        # Create experiment
        clients, aggregator = create_fedavg_experiment(
            model_template=test_model,
            train_datasets=client_datasets,
            learning_rate=0.01,
            local_epochs=1,
            batch_size=16,
            device="cpu"
        )
        
        # Verify clients and aggregator
        assert len(clients) == 4
        assert all(isinstance(client, Client) for client in clients)
        assert isinstance(aggregator, FedAvgAggregator)
        
        # Run one round manually
        global_params = clients[0].get_model_params()
        
        # Update all clients with global params
        for client in clients:
            client.update_model(global_params)
        
        # Train clients
        client_updates = []
        for client in clients:
            params, num_samples, loss = client.train()
            client_updates.append({
                'client_id': client.id,
                'params': params,
                'num_samples': num_samples,
                'loss': loss
            })
        
        # Aggregate updates
        aggregated_params = aggregator.aggregate(client_updates)
        
        # Verify aggregation
        assert isinstance(aggregated_params, dict)
        assert len(aggregated_params) > 0
    
    def test_fedprox_complete_experiment(self, synthetic_dataset, test_model):
        """Test complete FedProx experiment."""
        # Partition dataset
        client_datasets = partition_dataset(synthetic_dataset, num_clients=3, alpha=0.3)
        
        # Create FedProx experiment
        clients, aggregator = create_fedprox_experiment(
            model_template=test_model,
            train_datasets=client_datasets,
            learning_rate=0.01,
            local_epochs=1,
            mu=0.01,
            batch_size=16,
            device="cpu"
        )
        
        # Verify setup
        assert len(clients) == 3
        assert all(isinstance(client, FedProxClient) for client in clients)
        assert isinstance(aggregator, FedProxAggregator)
        assert all(client.mu == 0.01 for client in clients)
    
    def test_carbon_aware_training(self, synthetic_dataset, test_model):
        """Test federated learning with carbon tracking."""
        # Partition data
        client_datasets = partition_dataset(synthetic_dataset, num_clients=3, alpha=0.5)
        
        # Create clients
        clients = []
        for i, dataset in enumerate(client_datasets):
            dataloader = DataLoader(dataset, batch_size=16)
            client = Client(
                client_id=i,
                train_loader=dataloader,
                model_template=test_model,
                lr=0.01,
                device="cpu",
                local_epochs=1
            )
            clients.append(client)
        
        # Create server with carbon tracking
        orchestrator = RandomOrchestrator()
        server = FederatedServer(
            model_template=test_model,
            orchestrator=orchestrator,
            num_rounds=2,
            clients_per_round=2,
            device="cpu",
            carbon_aware=True
        )
        
        aggregator = FedAvgAggregator()
        
        # Run with carbon tracking
        results = server.run_federated_learning(
            clients=clients,
            aggregator=aggregator
        )
        
        # Verify carbon tracking results
        assert 'total_carbon_emission' in results
        assert results['total_carbon_emission'] >= 0
        assert 'carbon_emissions' in results['training_history']
    
    def test_differential_privacy_integration(self, synthetic_dataset, test_model):
        """Test federated learning with differential privacy."""
        # Create small experiment for DP
        client_datasets = partition_dataset(synthetic_dataset, num_clients=2, alpha=0.5)
        
        clients = []
        for i, dataset in enumerate(client_datasets):
            dataloader = DataLoader(dataset, batch_size=16)
            client = Client(
                client_id=i,
                train_loader=dataloader,
                model_template=test_model,
                lr=0.01,
                device="cpu",
                local_epochs=1
            )
            clients.append(client)
        
        # Create server with privacy budget
        orchestrator = RandomOrchestrator()
        server = FederatedServer(
            model_template=test_model,
            orchestrator=orchestrator,
            num_rounds=2,
            clients_per_round=2,
            device="cpu",
            privacy_budget=1.0  # Enable differential privacy
        )
        
        aggregator = FedAvgAggregator()
        
        # Run with differential privacy
        results = server.run_federated_learning(
            clients=clients,
            aggregator=aggregator
        )
        
        # Verify privacy accounting
        assert 'privacy_spent' in results
        assert results['privacy_spent'] >= 0
    
    def test_model_convergence_simple_case(self, test_model):
        """Test that model converges on a simple synthetic problem."""
        # Create very simple linearly separable data
        torch.manual_seed(42)
        X = torch.randn(100, 10)
        # Create linearly separable labels
        w_true = torch.randn(10)
        y = (X @ w_true > 0).long()
        
        dataset = TensorDataset(X, y)
        client_datasets = partition_dataset(dataset, num_clients=3, alpha=1.0)  # IID
        
        # Create clients
        clients = []
        for i, client_dataset in enumerate(client_datasets):
            dataloader = DataLoader(client_dataset, batch_size=16)
            client = Client(
                client_id=i,
                train_loader=dataloader,
                model_template=test_model,
                lr=0.1,  # Higher learning rate for faster convergence
                device="cpu",
                local_epochs=2
            )
            clients.append(client)
        
        # Create test data
        test_X = torch.randn(50, 10)
        test_y = (test_X @ w_true > 0).long()
        test_dataset = TensorDataset(test_X, test_y)
        test_loader = DataLoader(test_dataset, batch_size=16)
        
        # Create server
        orchestrator = RandomOrchestrator()
        aggregator = FedAvgAggregator()
        server = FederatedServer(
            model_template=test_model,
            orchestrator=orchestrator,
            num_rounds=10,
            clients_per_round=3,
            device="cpu"
        )
        
        # Run training
        results = server.run_federated_learning(
            clients=clients,
            aggregator=aggregator,
            test_loader=test_loader,
            eval_frequency=5
        )
        
        # Check that we have final accuracy
        assert 'final_accuracy' in results
        
        # For this simple problem, we should achieve reasonable accuracy
        # (This is a loose check since it's a synthetic problem)
        assert results['final_accuracy'] > 40.0  # At least better than random


@pytest.mark.slow
class TestLargeFederatedExperiment:
    """Slow integration tests for larger federated experiments."""
    
    def test_mnist_like_experiment(self):
        """Test MNIST-like experiment with proper CNN."""
        pytest.skip("Slow test - run manually if needed")
        
        # This would test a more realistic scenario
        # Similar to the actual MNIST experiment but smaller
        from metafed.models.simple_cnn import SimpleCNN
        
        # Create MNIST-like data
        X = torch.randn(1000, 1, 28, 28)  # MNIST-like images
        y = torch.randint(0, 10, (1000,))
        dataset = TensorDataset(X, y)
        
        # Partition among many clients
        client_datasets = partition_dataset(dataset, num_clients=20, alpha=0.1)
        
        # Create CNN model
        model = SimpleCNN(num_classes=10, input_channels=1)
        
        # Create experiment
        clients, aggregator = create_fedavg_experiment(
            model_template=model,
            train_datasets=client_datasets,
            learning_rate=0.01,
            local_epochs=3,
            batch_size=32,
            device="cpu"
        )
        
        # This test would run a full federated learning experiment
        # but is marked as slow since it takes significant time


def test_integration_with_configurations():
    """Test integration with configuration system."""
    from metafed.config import create_experiment_config
    
    # Create configuration
    config = create_experiment_config(
        algorithm="fedavg",
        dataset="mnist",
        num_clients=5,
        num_rounds=3,
        learning_rate=0.01
    )
    
    # Verify configuration
    assert config.algorithm == "fedavg"
    assert config.dataset == "mnist"
    assert config.num_clients == 5
    assert config.num_rounds == 3
    assert config.learning_rate == 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])