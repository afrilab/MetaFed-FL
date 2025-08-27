"""
CIFAR-10 Federated Learning Experiment Runner.

This script runs federated learning experiments on the CIFAR-10 dataset
with various algorithms, orchestration strategies, and privacy settings.
"""

import sys
import os
import argparse
import logging
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import Dict, List, Any, Optional
import time
import json

# Add src to path to import metafed modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from metafed.core.client import Client, FedProxClient, SCAFFOLDClient
from metafed.core.server import FederatedServer
from metafed.core.aggregation import FedAvgAggregator, FedProxAggregator, SCAFFOLDAggregator
from metafed.orchestration.random_orchestrator import RandomOrchestrator
from metafed.green.carbon_tracking import CarbonTracker
from metafed.utils.logging_config import setup_logging
from metafed.utils.metrics import compute_accuracy, plot_results
from metafed.data.loaders import create_federated_datasets
from metafed.models.simple_cnn import SimpleCNN


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CIFAR-10 Federated Learning Experiment")
    
    # General settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    
    # Federated learning settings
    parser.add_argument("--algorithm", type=str, default="fedprox", 
                       choices=["fedavg", "fedprox", "scaffold"],
                       help="Federated learning algorithm")
    parser.add_argument("--num-clients", type=int, default=50, help="Total number of clients")
    parser.add_argument("--clients-per-round", type=int, default=10, help="Clients per round")
    parser.add_argument("--num-rounds", type=int, default=100, help="Number of FL rounds")
    parser.add_argument("--local-epochs", type=int, default=5, help="Local training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    
    # Data settings
    parser.add_argument("--non-iid-alpha", type=float, default=0.5, help="Non-IID alpha parameter")
    
    # FedProx specific
    parser.add_argument("--fedprox-mu", type=float, default=0.01, help="FedProx mu parameter")
    
    # Orchestration
    parser.add_argument("--orchestrator", type=str, default="random", 
                       choices=["random", "rl"],
                       help="Client orchestration strategy")
    
    # Green computing
    parser.add_argument("--green-aware", action="store_true", help="Enable carbon-aware scheduling")
    parser.add_argument("--carbon-tracking", action="store_true", help="Enable carbon tracking")
    
    # Privacy
    parser.add_argument("--privacy", type=str, choices=["none", "differential"], 
                       default="none", help="Privacy mechanism")
    parser.add_argument("--epsilon", type=float, default=1.0, help="DP epsilon parameter")
    
    # Evaluation
    parser.add_argument("--eval-frequency", type=int, default=5, help="Evaluation frequency")
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found, using default settings")
        return {}


def setup_device(device_arg: str) -> str:
    """Setup computation device."""
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    
    logging.info(f"Using device: {device}")
    return device


def create_model() -> nn.Module:
    """Create model for CIFAR-10."""
    return SimpleCNN(num_classes=10, input_channels=3)


def create_clients(
    train_datasets: List[Subset],
    model_template: nn.Module,
    algorithm: str,
    lr: float,
    device: str,
    local_epochs: int,
    batch_size: int,
    fedprox_mu: float = 0.01
) -> List[Client]:
    """Create federated learning clients."""
    clients = []
    
    for i, dataset in enumerate(train_datasets):
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if algorithm == "fedavg":
            client = Client(
                client_id=i,
                train_loader=train_loader,
                model_template=model_template,
                lr=lr,
                device=device,
                local_epochs=local_epochs
            )
        elif algorithm == "fedprox":
            client = FedProxClient(
                client_id=i,
                train_loader=train_loader,
                model_template=model_template,
                lr=lr,
                device=device,
                local_epochs=local_epochs,
                mu=fedprox_mu
            )
        elif algorithm == "scaffold":
            client = SCAFFOLDClient(
                client_id=i,
                train_loader=train_loader,
                model_template=model_template,
                lr=lr,
                device=device,
                local_epochs=local_epochs
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        clients.append(client)
    
    logging.info(f"Created {len(clients)} {algorithm} clients")
    return clients


def create_aggregator(algorithm: str):
    """Create aggregator based on algorithm."""
    if algorithm == "fedavg":
        return FedAvgAggregator()
    elif algorithm == "fedprox":
        return FedProxAggregator()
    elif algorithm == "scaffold":
        return SCAFFOLDAggregator()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def create_orchestrator(orchestrator_type: str):
    """Create client orchestrator."""
    if orchestrator_type == "random":
        return RandomOrchestrator()
    elif orchestrator_type == "rl":
        # Placeholder for RL orchestrator
        logging.warning("RL orchestrator not implemented, using random")
        return RandomOrchestrator()
    else:
        raise ValueError(f"Unknown orchestrator: {orchestrator_type}")


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """Run federated learning experiment."""
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = setup_device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(os.path.join(args.output_dir, "experiment.log"))
    
    logging.info("Starting CIFAR-10 federated learning experiment")
    logging.info(f"Configuration: {vars(args)}")
    
    # Load data
    logging.info("Loading and partitioning CIFAR-10 dataset")
    train_datasets, test_loader = create_federated_datasets(
        dataset_name="cifar10",
        num_clients=args.num_clients,
        non_iid_alpha=args.non_iid_alpha,
        batch_size=args.batch_size
    )
    
    # Create model
    model_template = create_model()
    logging.info(f"Created model: {model_template.__class__.__name__}")
    
    # Create clients
    clients = create_clients(
        train_datasets=train_datasets,
        model_template=model_template,
        algorithm=args.algorithm,
        lr=args.learning_rate,
        device=device,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        fedprox_mu=args.fedprox_mu
    )
    
    # Create aggregator
    aggregator = create_aggregator(args.algorithm)
    
    # Create orchestrator
    orchestrator = create_orchestrator(args.orchestrator)
    
    # Create server
    server = FederatedServer(
        model_template=model_template,
        orchestrator=orchestrator,
        num_rounds=args.num_rounds,
        clients_per_round=args.clients_per_round,
        device=device,
        carbon_aware=args.green_aware,
        privacy_budget=args.epsilon if args.privacy == "differential" else None
    )
    
    # Run federated learning
    logging.info("Starting federated learning training")
    start_time = time.time()
    
    results = server.run_federated_learning(
        clients=clients,
        aggregator=aggregator,
        test_loader=test_loader,
        eval_frequency=args.eval_frequency
    )
    
    training_time = time.time() - start_time
    results["total_training_time"] = training_time
    
    logging.info(f"Experiment completed in {training_time:.2f} seconds")
    
    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, 'w') as f:
        # Convert tensors to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                json_results[key] = value.tolist()
            elif key == "final_model_state":
                # Skip model state dict for JSON
                continue
            else:
                json_results[key] = value
        
        json.dump(json_results, f, indent=2)
    
    logging.info(f"Results saved to {results_path}")
    
    # Plot results if matplotlib is available
    try:
        plot_results(results, save_path=os.path.join(args.output_dir, "plots.png"))
        logging.info("Plots saved")
    except ImportError:
        logging.warning("Matplotlib not available, skipping plots")
    
    return results


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        # Update args with config values (command line takes precedence)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    try:
        results = run_experiment(args)
        
        # Print summary
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        print(f"Algorithm: {args.algorithm}")
        print(f"Rounds: {args.num_rounds}")
        print(f"Clients: {args.num_clients} (per round: {args.clients_per_round})")
        
        if "final_accuracy" in results:
            print(f"Final Accuracy: {results['final_accuracy']:.2f}%")
        
        if "total_carbon_emission" in results:
            print(f"Carbon Emission: {results['total_carbon_emission']:.6f} kg CO2")
        
        print(f"Training Time: {results['total_training_time']:.2f} seconds")
        print("="*50)
        
    except Exception as e:
        logging.error(f"Experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()