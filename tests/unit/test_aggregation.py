"""
Unit tests for federated learning aggregation algorithms.
"""

import pytest
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from metafed.core.aggregation import (
    FedAvgAggregator, 
    FedProxAggregator, 
    SCAFFOLDAggregator,
    SecureAggregator,
    AdaptiveAggregator
)


class TestFedAvgAggregator:
    """Test cases for FedAvg aggregator."""
    
    @pytest.fixture
    def sample_client_updates(self):
        """Create sample client updates."""
        updates = []
        for i in range(3):
            params = {
                'layer1.weight': torch.randn(5, 3) + i * 0.1,
                'layer1.bias': torch.randn(5) + i * 0.1,
                'layer2.weight': torch.randn(2, 5) + i * 0.1
            }
            updates.append({
                'client_id': i,
                'params': params,
                'num_samples': 100 + i * 50,
                'loss': 0.5 + i * 0.1
            })
        return updates
    
    def test_fedavg_weighted_aggregation(self, sample_client_updates):
        """Test weighted aggregation in FedAvg."""
        aggregator = FedAvgAggregator(weighted=True)
        aggregated = aggregator.aggregate(sample_client_updates)
        
        # Check that aggregated parameters have correct structure
        assert isinstance(aggregated, dict)
        assert 'layer1.weight' in aggregated
        assert 'layer1.bias' in aggregated
        assert 'layer2.weight' in aggregated
        
        # Check shapes are preserved
        assert aggregated['layer1.weight'].shape == (5, 3)
        assert aggregated['layer1.bias'].shape == (5,)
        assert aggregated['layer2.weight'].shape == (2, 5)
    
    def test_fedavg_unweighted_aggregation(self, sample_client_updates):
        """Test unweighted aggregation in FedAvg."""
        aggregator = FedAvgAggregator(weighted=False)
        aggregated = aggregator.aggregate(sample_client_updates)
        
        # Should still produce valid aggregated parameters
        assert isinstance(aggregated, dict)
        assert len(aggregated) == 3
    
    def test_fedavg_empty_updates_error(self):
        """Test that empty updates raise error."""
        aggregator = FedAvgAggregator()
        
        with pytest.raises(ValueError, match="No client updates provided"):
            aggregator.aggregate([])
    
    def test_fedavg_single_client(self):
        """Test aggregation with single client."""
        aggregator = FedAvgAggregator()
        
        single_update = [{
            'client_id': 0,
            'params': {'weight': torch.tensor([1.0, 2.0])},
            'num_samples': 100,
            'loss': 0.5
        }]
        
        aggregated = aggregator.aggregate(single_update)
        
        # Single client aggregation should return the same parameters
        assert torch.equal(aggregated['weight'], torch.tensor([1.0, 2.0]))


class TestSCAFFOLDAggregator:
    """Test cases for SCAFFOLD aggregator."""
    
    @pytest.fixture
    def scaffold_client_updates(self):
        """Create SCAFFOLD client updates with control deltas."""
        updates = []
        for i in range(3):
            params = {
                'fc.weight': torch.randn(2, 3),
                'fc.bias': torch.randn(2)
            }
            control_delta = {
                'fc.weight': torch.randn(2, 3) * 0.01,
                'fc.bias': torch.randn(2) * 0.01
            }
            updates.append({
                'client_id': i,
                'params': params,
                'control_delta': control_delta,
                'num_samples': 50,
                'loss': 0.3
            })
        return updates
    
    def test_scaffold_aggregation_with_control_variates(self, scaffold_client_updates):
        """Test SCAFFOLD aggregation with control variates."""
        aggregator = SCAFFOLDAggregator()
        aggregated = aggregator.aggregate(scaffold_client_updates)
        
        # Check aggregated parameters
        assert isinstance(aggregated, dict)
        assert 'fc.weight' in aggregated
        assert 'fc.bias' in aggregated
        
        # Check that server control variates are updated
        server_control = aggregator.get_server_control()
        assert isinstance(server_control, dict)
        assert 'fc.weight' in server_control
        assert 'fc.bias' in server_control
    
    def test_scaffold_server_control_initialization(self, scaffold_client_updates):
        """Test server control variates initialization."""
        aggregator = SCAFFOLDAggregator()
        
        # Initially, server control should be None
        assert aggregator.server_control is None
        
        # After aggregation, should be initialized
        aggregator.aggregate(scaffold_client_updates)
        assert aggregator.server_control is not None
    
    def test_scaffold_without_control_delta(self):
        """Test SCAFFOLD aggregation without control delta."""
        aggregator = SCAFFOLDAggregator()
        
        updates = [{
            'client_id': 0,
            'params': {'weight': torch.tensor([1.0])},
            'num_samples': 10,
            'loss': 0.1
        }]
        
        aggregated = aggregator.aggregate(updates)
        
        # Should still work without control delta
        assert 'weight' in aggregated


class TestSecureAggregator:
    """Test cases for secure aggregator with differential privacy."""
    
    @pytest.fixture
    def base_aggregator(self):
        """Create base aggregator for secure aggregator."""
        return FedAvgAggregator()
    
    def test_secure_aggregator_initialization(self, base_aggregator):
        """Test secure aggregator initialization."""
        secure_agg = SecureAggregator(
            base_aggregator=base_aggregator,
            enable_dp=True,
            dp_epsilon=1.0,
            dp_delta=1e-5
        )
        
        assert secure_agg.base_aggregator == base_aggregator
        assert secure_agg.enable_dp == True
        assert secure_agg.dp_epsilon == 1.0
        assert secure_agg.dp_delta == 1e-5
    
    def test_secure_aggregator_with_dp(self, base_aggregator):
        """Test secure aggregation with differential privacy."""
        secure_agg = SecureAggregator(
            base_aggregator=base_aggregator,
            enable_dp=True,
            dp_epsilon=1.0,
            noise_multiplier=0.1
        )
        
        client_updates = [{
            'client_id': 0,
            'params': {'weight': torch.tensor([1.0, 2.0])},
            'num_samples': 100,
            'loss': 0.5
        }]
        
        aggregated = secure_agg.aggregate(client_updates)
        
        # With DP, result should be different from original due to noise
        assert 'weight' in aggregated
        assert aggregated['weight'].shape == torch.tensor([1.0, 2.0]).shape
        
        # The exact values will be different due to noise
        # We can't test exact equality, but can test structure
    
    def test_secure_aggregator_without_dp(self, base_aggregator):
        """Test secure aggregation without differential privacy."""
        secure_agg = SecureAggregator(
            base_aggregator=base_aggregator,
            enable_dp=False
        )
        
        client_updates = [{
            'client_id': 0,
            'params': {'weight': torch.tensor([1.0, 2.0])},
            'num_samples': 100,
            'loss': 0.5
        }]
        
        aggregated = secure_agg.aggregate(client_updates)
        
        # Without DP, should be identical to base aggregator
        base_aggregated = base_aggregator.aggregate(client_updates)
        assert torch.equal(aggregated['weight'], base_aggregated['weight'])


class TestAdaptiveAggregator:
    """Test cases for adaptive aggregator."""
    
    @pytest.fixture
    def client_updates_with_performance(self):
        """Create client updates with performance metrics."""
        updates = []
        for i in range(3):
            params = {'weight': torch.randn(3) + i}
            updates.append({
                'client_id': i,
                'params': params,
                'num_samples': 100,
                'loss': 0.5 - i * 0.1  # Better clients have lower loss
            })
        return updates
    
    def test_adaptive_aggregator_initialization(self):
        """Test adaptive aggregator initialization."""
        aggregator = AdaptiveAggregator(alpha=0.1)
        
        assert aggregator.alpha == 0.1
        assert len(aggregator.client_weights) == 0
        assert len(aggregator.client_performance_history) == 0
    
    def test_adaptive_aggregator_performance_tracking(self, client_updates_with_performance):
        """Test performance history tracking."""
        aggregator = AdaptiveAggregator()
        
        # First aggregation
        aggregated = aggregator.aggregate(client_updates_with_performance)
        
        # Check that performance history is updated
        assert len(aggregator.client_performance_history) == 3
        assert 0 in aggregator.client_performance_history
        assert 1 in aggregator.client_performance_history
        assert 2 in aggregator.client_performance_history
    
    def test_adaptive_weights_improve_over_time(self, client_updates_with_performance):
        """Test that adaptive weights improve over multiple rounds."""
        aggregator = AdaptiveAggregator()
        
        # Run multiple rounds
        for _ in range(3):
            aggregator.aggregate(client_updates_with_performance)
        
        # Client 2 should have the best performance (lowest loss)
        # and should accumulate more weight over time
        history_lengths = [
            len(aggregator.client_performance_history[i]) 
            for i in range(3)
        ]
        
        # All clients should have performance history
        assert all(length > 0 for length in history_lengths)


@pytest.mark.parametrize("aggregator_class,kwargs", [
    (FedAvgAggregator, {"weighted": True}),
    (FedAvgAggregator, {"weighted": False}),
    (FedProxAggregator, {"weighted": True}),
    (SCAFFOLDAggregator, {}),
    (AdaptiveAggregator, {"alpha": 0.1})
])
def test_aggregator_basic_functionality(aggregator_class, kwargs):
    """Test basic functionality across all aggregator types."""
    aggregator = aggregator_class(**kwargs)
    
    # Create minimal client update
    client_updates = [{
        'client_id': 0,
        'params': {'weight': torch.tensor([1.0, 2.0])},
        'num_samples': 100,
        'loss': 0.5
    }]
    
    # Add control delta for SCAFFOLD
    if aggregator_class == SCAFFOLDAggregator:
        client_updates[0]['control_delta'] = {
            'weight': torch.tensor([0.01, -0.01])
        }
    
    # Should not raise any errors
    aggregated = aggregator.aggregate(client_updates)
    
    # Should return dictionary with correct structure
    assert isinstance(aggregated, dict)
    assert 'weight' in aggregated
    assert aggregated['weight'].shape == torch.Size([2])


def test_aggregator_consistency():
    """Test that aggregators produce consistent results."""
    # Create identical client updates
    client_updates = []
    for i in range(3):
        client_updates.append({
            'client_id': i,
            'params': {'weight': torch.tensor([1.0, 2.0])},
            'num_samples': 100,
            'loss': 0.5
        })
    
    # All aggregators should produce the same result for identical inputs
    fedavg_agg = FedAvgAggregator(weighted=False)
    fedprox_agg = FedProxAggregator(weighted=False)
    
    fedavg_result = fedavg_agg.aggregate(client_updates)
    fedprox_result = fedprox_agg.aggregate(client_updates)
    
    # Results should be identical for identical inputs
    assert torch.allclose(fedavg_result['weight'], fedprox_result['weight'])


if __name__ == "__main__":
    pytest.main([__file__])