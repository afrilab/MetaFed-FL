"""
Simple CNN model for MNIST and CIFAR-10 experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN model suitable for MNIST and CIFAR-10.
    
    This is a lightweight CNN with good performance for
    federated learning experiments.
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 1):
        """
        Initialize SimpleCNN.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (1 for MNIST, 3 for CIFAR-10)
        """
        super(SimpleCNN, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.25)
        self.dropout_fc = nn.Dropout(0.5)
        
        # Calculate the size of flattened features
        # For MNIST (28x28) or CIFAR-10 (32x32), after 3 pooling operations
        if input_channels == 1:  # MNIST
            self.feature_size = 128 * 3 * 3  # 28 -> 14 -> 7 -> 3 (with padding)
        else:  # CIFAR-10
            self.feature_size = 128 * 4 * 4  # 32 -> 16 -> 8 -> 4
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Third convolutional block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        
        x = self.fc3(x)
        
        return x
    
    def get_feature_size(self) -> int:
        """Get the size of features before the classifier."""
        return self.feature_size


class LeNet(nn.Module):
    """
    LeNet-5 architecture for MNIST.
    
    Classic CNN architecture suitable for MNIST experiments.
    """
    
    def __init__(self, num_classes: int = 10):
        """
        Initialize LeNet.
        
        Args:
            num_classes: Number of output classes
        """
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        self.pool = nn.AvgPool2d(2, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_model(model_name: str, num_classes: int = 10, input_channels: int = 1) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_name: Name of the model
        num_classes: Number of output classes
        input_channels: Number of input channels
        
    Returns:
        Model instance
    """
    if model_name.lower() == "simplecnn":
        return SimpleCNN(num_classes=num_classes, input_channels=input_channels)
    elif model_name.lower() == "lenet":
        if input_channels != 1:
            raise ValueError("LeNet only supports single channel input")
        return LeNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# Model registry for easy access
MODEL_REGISTRY = {
    "simplecnn": SimpleCNN,
    "lenet": LeNet
}