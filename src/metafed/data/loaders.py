"""
Data loading and partitioning utilities for federated learning.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def create_federated_datasets(
    dataset_name: str,
    num_clients: int,
    non_iid_alpha: float = 0.5,
    batch_size: int = 32,
    data_dir: str = "./data"
) -> Tuple[List[Subset], DataLoader]:
    """
    Create federated datasets with non-IID partitioning.
    
    Args:
        dataset_name: Name of dataset ("mnist" or "cifar10")
        num_clients: Number of clients
        non_iid_alpha: Non-IID parameter (lower = more non-IID)
        batch_size: Batch size for test loader
        data_dir: Directory to store datasets
        
    Returns:
        Tuple of (train_datasets, test_loader)
    """
    
    # Define transforms
    if dataset_name.lower() == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load MNIST
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
        
    elif dataset_name.lower() == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Load CIFAR-10
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform_test)
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Partition training data among clients
    train_datasets = partition_dataset(train_dataset, num_clients, non_iid_alpha)
    
    logger.info(f"Created federated {dataset_name} datasets for {num_clients} clients")
    logger.info(f"Non-IID alpha: {non_iid_alpha}")
    
    return train_datasets, test_loader


def partition_dataset(
    dataset: torch.utils.data.Dataset,
    num_clients: int,
    alpha: float = 0.5
) -> List[Subset]:
    """
    Partition dataset among clients using Dirichlet distribution.
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
        
    Returns:
        List of dataset subsets for each client
    """
    
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        # Extract labels manually
        labels = []
        for _, label in dataset:
            labels.append(label)
        labels = np.array(labels)
    
    num_classes = len(np.unique(labels))
    num_samples = len(labels)
    
    # Sort indices by label
    sorted_indices = np.argsort(labels)
    
    # Split by class
    class_indices = []
    for class_id in range(num_classes):
        class_mask = labels == class_id
        class_indices.append(np.where(class_mask)[0])
    
    # Distribute samples using Dirichlet distribution
    client_indices = [[] for _ in range(num_clients)]
    
    for class_id in range(num_classes):
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Calculate number of samples for each client
        class_samples = len(class_indices[class_id])
        client_samples = (proportions * class_samples).astype(int)
        
        # Adjust for rounding errors
        client_samples[-1] = class_samples - np.sum(client_samples[:-1])
        
        # Distribute samples
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + client_samples[client_id]
            client_indices[client_id].extend(
                class_indices[class_id][start_idx:end_idx]
            )
            start_idx = end_idx
    
    # Create subsets
    client_datasets = []
    for client_id in range(num_clients):
        indices = client_indices[client_id]
        np.random.shuffle(indices)  # Shuffle client's samples
        client_datasets.append(Subset(dataset, indices))
    
    # Log distribution statistics
    client_sizes = [len(subset) for subset in client_datasets]
    logger.info(f"Client dataset sizes - Min: {min(client_sizes)}, "
               f"Max: {max(client_sizes)}, Avg: {np.mean(client_sizes):.1f}")
    
    return client_datasets


def create_iid_partition(
    dataset: torch.utils.data.Dataset,
    num_clients: int
) -> List[Subset]:
    """
    Create IID partition of dataset.
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        
    Returns:
        List of dataset subsets for each client
    """
    num_samples = len(dataset)
    samples_per_client = num_samples // num_clients
    
    # Create random permutation of indices
    indices = np.random.permutation(num_samples)
    
    client_datasets = []
    for client_id in range(num_clients):
        start_idx = client_id * samples_per_client
        if client_id == num_clients - 1:
            # Last client gets remaining samples
            end_idx = num_samples
        else:
            end_idx = start_idx + samples_per_client
        
        client_indices = indices[start_idx:end_idx]
        client_datasets.append(Subset(dataset, client_indices))
    
    return client_datasets


def analyze_data_distribution(client_datasets: List[Subset]) -> dict:
    """
    Analyze the distribution of labels across clients.
    
    Args:
        client_datasets: List of client datasets
        
    Returns:
        Dictionary with distribution statistics
    """
    client_label_counts = []
    
    for dataset in client_datasets:
        label_counts = {}
        for idx in dataset.indices:
            _, label = dataset.dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            label_counts[label] = label_counts.get(label, 0) + 1
        client_label_counts.append(label_counts)
    
    # Calculate statistics
    all_labels = set()
    for counts in client_label_counts:
        all_labels.update(counts.keys())
    
    num_classes = len(all_labels)
    
    # Calculate average number of classes per client
    classes_per_client = [len(counts) for counts in client_label_counts]
    avg_classes_per_client = np.mean(classes_per_client)
    
    # Calculate Herfindahl index (measure of concentration)
    herfindahl_indices = []
    for counts in client_label_counts:
        total_samples = sum(counts.values())
        proportions = [count / total_samples for count in counts.values()]
        herfindahl = sum(p**2 for p in proportions)
        herfindahl_indices.append(herfindahl)
    
    avg_herfindahl = np.mean(herfindahl_indices)
    
    return {
        "num_clients": len(client_datasets),
        "num_classes": num_classes,
        "avg_classes_per_client": avg_classes_per_client,
        "min_classes_per_client": min(classes_per_client),
        "max_classes_per_client": max(classes_per_client),
        "avg_herfindahl_index": avg_herfindahl,
        "client_label_counts": client_label_counts
    }