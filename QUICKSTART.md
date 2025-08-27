# MetaFed-FL Quick Start Guide

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/afrilab/MetaFed-FL.git
cd MetaFed-FL
```

### 2. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .

# For development with all tools
pip install -r requirements-dev.txt
```

### 3. Install the Package
```bash
# Install as editable package
pip install -e .
```

## Quick Start

### Running MNIST Experiment
```bash
# Simple MNIST experiment with default settings
python -m experiments.mnist.run_experiment

# With custom parameters
python -m experiments.mnist.run_experiment \
    --algorithm fedavg \
    --num-rounds 50 \
    --clients-per-round 5 \
    --learning-rate 0.01

# Using configuration file
python -m experiments.mnist.run_experiment \
    --config experiments/configs/mnist_fedavg.yaml
```

### Running CIFAR-10 Experiment
```bash
# CIFAR-10 with FedProx
python -m experiments.cifar10.run_experiment \
    --algorithm fedprox \
    --fedprox-mu 0.01 \
    --num-rounds 100

# With privacy and green computing
python -m experiments.mnist.run_experiment \
    --privacy differential \
    --epsilon 1.0 \
    --green-aware \
    --carbon-tracking
```

### Using the Package Programmatically

```python
import torch
from metafed.core.client import Client
from metafed.core.server import FederatedServer
from metafed.core.aggregation import FedAvgAggregator
from metafed.orchestration.random_orchestrator import RandomOrchestrator
from metafed.models.simple_cnn import ResNet18
from metafed.data.loaders import create_federated_datasets

# Create model
model = ResNet18(num_classes=10, input_channels=1)

# Load data
train_datasets, test_loader = create_federated_datasets(
    dataset_name="mnist",
    num_clients=10,
    non_iid_alpha=0.5
)

# Create clients
clients = []
for i, dataset in enumerate(train_datasets):
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    client = Client(
        client_id=i,
        train_loader=train_loader,
        model_template=model,
        lr=0.01
    )
    clients.append(client)

# Create server components
orchestrator = RandomOrchestrator()
aggregator = FedAvgAggregator()
server = FederatedServer(
    model_template=model,
    orchestrator=orchestrator,
    num_rounds=50,
    clients_per_round=5
)

# Run federated learning
results = server.run_federated_learning(
    clients=clients,
    aggregator=aggregator,
    test_loader=test_loader
)

print(f"Final accuracy: {results['final_accuracy']:.2f}%")
```

## Configuration Files

Configuration files are located in `experiments/configs/`:
- `mnist_fedavg.yaml` - MNIST with FedAvg
- `cifar10_fedprox.yaml` - CIFAR-10 with FedProx

Example configuration structure:
```yaml
# General settings
seed: 42
algorithm: "fedavg"
num_clients: 50
num_rounds: 100

# Privacy and green computing
privacy: "differential"
epsilon: 1.0
green_aware: true
carbon_tracking: true
```

## Available Algorithms

- **FedAvg**: Standard federated averaging
- **FedProx**: Federated optimization with proximal term
- **SCAFFOLD**: Stochastic controlled averaging with control variates

## Available Features

### 🔒 Privacy Preservation
- Differential privacy with configurable ε
- Secure aggregation (planned)
- Homomorphic encryption (planned)

### 🌱 Green Computing
- Carbon emission tracking
- Carbon-aware client scheduling
- Renewable energy alignment (planned)

### 🤖 Orchestration
- Random client selection
- RL-based orchestration (planned)
- Performance-based selection (planned)

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

### Building Documentation
```bash
cd docs/
make html
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you've installed the package with `pip install -e .`
2. **CUDA Errors**: Set `--device cpu` if CUDA is not available
3. **Memory Issues**: Reduce `--batch-size` or `--clients-per-round`

### Getting Help

- Check the documentation
- Open an issue on GitHub
- Review the example configurations

## Migration from Notebooks

The original Jupyter notebooks (`metafed-mnist.ipynb`, `metafed-cifar.ipynb`) have been converted to:

- **Modular code**: `src/metafed/` package
- **Experiment runners**: `experiments/mnist/run_experiment.py`
- **Configuration files**: `experiments/configs/`
- **Proper testing**: `tests/` directory

The notebooks are preserved for reference but the new modular structure is recommended for all development and experimentation.