"""
Test suite for MetaFed-FL.

This package contains comprehensive tests for all MetaFed-FL components
including unit tests, integration tests, and benchmarks.
"""

import pytest
import torch
import logging

# Setup test logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise in tests

# Common test fixtures and utilities
@pytest.fixture
def device():
    """Get computation device for tests."""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    import torch.nn as nn
    
    class SimpleTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 2)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    return SimpleTestModel()

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    
    # Generate random data
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    return dataloader

# Test configuration
TEST_CONFIG = {
    "fast_tests": True,  # Skip slow tests by default
    "integration_tests": False,  # Skip integration tests by default
    "mock_external_services": True,  # Mock external services
}