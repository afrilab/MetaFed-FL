# MetaFed-FL: Federated Learning for Metaverse Systems

<div align="center">

<!-- Status and Quality Badges -->
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-orange)
![Code Quality](https://img.shields.io/badge/code%20quality-A-green)

<!-- Project Information Badges -->
![License](https://img.shields.io/badge/license-MIT-blue)
![Version](https://img.shields.io/badge/version-1.0.0-green)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-lightgrey)

<!-- Research and Academic Badges -->
![Paper](https://img.shields.io/badge/paper-arXiv-red)
![Conference](https://img.shields.io/badge/conference-pending-yellow)
![Framework](https://img.shields.io/badge/framework-federated%20learning-purple)

</div>

---

## 🔬 Research Overview

**MetaFed** is a cutting-edge federated learning (FL) framework specifically designed for **Metaverse infrastructures**. This research addresses the critical challenges of privacy, performance, and sustainability in distributed AI systems.

**Paper:** *MetaFed: Advancing Privacy, Performance, and Sustainability in Federated Metaverse Systems*  
**Authors:** Muhammet Anil Yagiz, Zeynep Sude Cengiz, Polat Goktas (2025)

---

## 🛠️ Installation

### Prerequisites

- **Python 3.9+**
- **PyTorch 2.2.2+**
- **CUDA** (optional, for GPU acceleration)

### Dependencies

```bash
# Core dependencies
pip install torch>=2.2.2 torchvision>=0.17.2
pip install numpy>=1.26.4 pandas>=2.2.2
pip install matplotlib>=3.9.2 timm>=1.0.8

# Development dependencies
pip install -r requirements-dev.txt
```

### Package Installation

```bash
# Install as editable package
pip install -e .

# Or install from PyPI (when available)
pip install metafed-fl
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/username/MetaFed-FL.git
cd MetaFed-FL

# Install dependencies
pip install -r requirements.txt

# Run MNIST experiment
python -m experiments.mnist.run_experiment

# Run CIFAR-10 experiment
python -m experiments.cifar10.run_experiment
```

---

## 📊 Key Features

🤖 **Multi-Agent Reinforcement Learning (MARL)**
- Dynamic client orchestration and selection
- Adaptive resource allocation
- Intelligent scheduling algorithms

🔒 **Privacy-Preserving Techniques**
- Homomorphic encryption for secure aggregation
- Differential privacy for data protection
- Zero-knowledge proof mechanisms

🌱 **Carbon-Aware Scheduling**
- Real-time carbon intensity tracking
- Renewable energy-aligned orchestration
- Sustainable resource management

📈 **Comprehensive Benchmarks**
- MNIST and CIFAR-10 datasets
- Multiple federated learning algorithms
- Extensive performance metrics

---

## 📈 Performance Metrics

| Dataset | Algorithm | Accuracy Improvement | CO2 Reduction | Efficiency Gain |
|---------|-----------|---------------------|---------------|----------------|
| MNIST   | FedAvg+MARL | +15% | -35% | +20% |
| CIFAR-10| FedProx+MARL| +20% | -45% | +25% |
| CIFAR-10| SCAFFOLD+MARL| +18% | -40% | +22% |

*Compared to baseline federated learning algorithms*

---

## 📂 Project Structure

```
MetaFed-FL/
├── src/metafed/              # Core package
│   ├── core/                 # FL core components
│   ├── algorithms/           # FL algorithms
│   ├── orchestration/        # MARL orchestration
│   ├── privacy/              # Privacy mechanisms
│   ├── green/                # Carbon-aware features
│   └── utils/                # Utilities
├── experiments/              # Experiment runners
│   ├── mnist/                # MNIST experiments
│   └── cifar10/              # CIFAR-10 experiments
├── tests/                    # Test suite
├── docs/                     # Documentation
└── scripts/                  # Utility scripts
```

---

## 🧪 Running Experiments

### Command Line Interface

```bash
# MNIST with FedAvg + MARL
metafed-mnist --algorithm fedavg --orchestrator rl --rounds 100

# CIFAR-10 with privacy preservation
metafed-cifar10 --algorithm fedprox --privacy differential --epsilon 1.0

# Green-aware scheduling
metafed-mnist --green-aware --carbon-tracking
```

### Configuration Files

```bash
# Using YAML configuration
python -m experiments.mnist.run_experiment --config configs/fedavg_privacy.yaml
```

### Jupyter Notebooks (Legacy)

```bash
# For research and exploration
jupyter notebook experiments/notebooks/
```

---

## 🔧 Advanced Configuration

### Algorithm Selection

- **FedAvg**: Standard federated averaging
- **FedProx**: Proximal federated optimization
- **SCAFFOLD**: Stochastic controlled averaging

### Orchestration Methods

- **Random**: Random client selection
- **RL-based**: Multi-agent reinforcement learning
- **Green-aware**: Carbon-optimized selection

### Privacy Settings

- **Homomorphic Encryption**: Secure aggregation
- **Differential Privacy**: ε-differential privacy
- **Hybrid**: Combined privacy mechanisms

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/username/MetaFed-FL.git
cd MetaFed-FL
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
flake8 src/ tests/
black src/ tests/
```

### Code Style

- **Black** for code formatting
- **Flake8** for linting
- **isort** for import sorting
- **Type hints** for better code documentation

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📝 Citation

If you use MetaFed-FL in your research, please cite:

```bibtex
@misc{yagiz2025metafedadvancingprivacyperformance,
  title={MetaFed: Advancing Privacy, Performance, and Sustainability in Federated Metaverse Systems},
  author={Muhammet Anil Yagiz and Zeynep Sude Cengiz and Polat Goktas},
  year={2025},
  eprint={2508.17341},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2508.17341}
}
```

---

## 🌟 Acknowledgments

- **AFRI Lab** for supporting this research
- **PyTorch Team** for the excellent deep learning framework
- **Federated Learning Community** for inspiration and collaboration

---

<div align="center">

**[🌐 Website](https://metafed-fl.org)** • **[📖 Docs](https://metafed-fl.readthedocs.io)** • **[💬 Discussions](https://github.com/username/MetaFed-FL/discussions)** • **[🐛 Issues](https://github.com/username/MetaFed-FL/issues)**

*Built with ❤️ for the Federated Learning Community*

</div>