"""
Simple CNN model for MNIST and CIFAR-10 experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Union, List, Optional


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


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet implementation for CIFAR-10.
    
    This is a modified ResNet that works well with CIFAR-10's 32x32 images.
    Commonly used in federated learning research.
    """
    
    def __init__(self, block: Type[BasicBlock], num_blocks: List[int], num_classes: int = 10, input_channels: int = 3):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.input_channels = input_channels
        
        # Initial convolution (smaller kernel for CIFAR-10)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Final classifier
        self.linear = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block: Type[BasicBlock], planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes: int = 10, input_channels: int = 3) -> ResNet:
    """
    ResNet-18 model for CIFAR-10.
    
    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels
        
    Returns:
        ResNet18 model instance
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, input_channels=input_channels)


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
    elif model_name.lower() == "resnet18":
        return ResNet18(num_classes=num_classes, input_channels=input_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# Model registry for easy access
MODEL_REGISTRY = {
    "simplecnn": SimpleCNN,
    "lenet": LeNet,
    "resnet18": ResNet18
}