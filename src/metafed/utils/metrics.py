"""
Metrics and evaluation utilities for MetaFed-FL.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_accuracy(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: str) -> float:
    """
    Compute accuracy of model on given data.
    
    Args:
        model: Neural network model
        data_loader: Data loader
        device: Computation device
        
    Returns:
        Accuracy as percentage
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy


def compute_loss(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: str) -> float:
    """
    Compute average loss of model on given data.
    
    Args:
        model: Neural network model
        data_loader: Data loader
        device: Computation device
        
    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss


def plot_results(results: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Plot training results.
    
    Args:
        results: Results dictionary from federated learning
        save_path: Path to save plot (optional)
    """
    try:
        history = results.get('training_history', {})
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Federated Learning Results', fontsize=16)
        
        # Accuracy plot
        if 'accuracies' in history and history['accuracies']:
            axes[0, 0].plot(history['rounds'][::len(history['rounds'])//len(history['accuracies'])], 
                           history['accuracies'], 'b-', marker='o')
            axes[0, 0].set_title('Test Accuracy')
            axes[0, 0].set_xlabel('Round')
            axes[0, 0].set_ylabel('Accuracy (%)')
            axes[0, 0].grid(True)
        
        # Loss plot
        if 'losses' in history and history['losses']:
            axes[0, 1].plot(history['rounds'], history['losses'], 'r-', marker='s')
            axes[0, 1].set_title('Training Loss')
            axes[0, 1].set_xlabel('Round')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)
        
        # Carbon emissions plot
        if 'carbon_emissions' in history and history['carbon_emissions']:
            axes[1, 0].plot(history['rounds'], history['carbon_emissions'], 'g-', marker='^')
            axes[1, 0].set_title('Carbon Emissions per Round')
            axes[1, 0].set_xlabel('Round')
            axes[1, 0].set_ylabel('CO2 (kg)')
            axes[1, 0].grid(True)
        
        # Training time plot
        if 'training_times' in history and history['training_times']:
            axes[1, 1].plot(history['rounds'], history['training_times'], 'm-', marker='d')
            axes[1, 1].set_title('Training Time per Round')
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Time (s)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
            
    except Exception as e:
        logger.error(f"Error plotting results: {e}")


def calculate_improvement_metrics(baseline_results: Dict[str, Any], 
                                new_results: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate improvement metrics compared to baseline.
    
    Args:
        baseline_results: Baseline experiment results
        new_results: New experiment results
        
    Returns:
        Dictionary with improvement metrics
    """
    improvements = {}
    
    # Accuracy improvement
    if 'final_accuracy' in baseline_results and 'final_accuracy' in new_results:
        baseline_acc = baseline_results['final_accuracy']
        new_acc = new_results['final_accuracy']
        improvements['accuracy_improvement_percent'] = ((new_acc - baseline_acc) / baseline_acc) * 100
    
    # Carbon reduction
    if 'total_carbon_emission' in baseline_results and 'total_carbon_emission' in new_results:
        baseline_carbon = baseline_results['total_carbon_emission']
        new_carbon = new_results['total_carbon_emission']
        improvements['carbon_reduction_percent'] = ((baseline_carbon - new_carbon) / baseline_carbon) * 100
    
    # Training time improvement
    if 'total_training_time' in baseline_results and 'total_training_time' in new_results:
        baseline_time = baseline_results['total_training_time']
        new_time = new_results['total_training_time']
        improvements['time_improvement_percent'] = ((baseline_time - new_time) / baseline_time) * 100
    
    return improvements


def generate_summary_report(results: Dict[str, Any]) -> str:
    """
    Generate a summary report of the experiment results.
    
    Args:
        results: Experiment results
        
    Returns:
        Formatted summary report string
    """
    report = []
    report.append("="*60)
    report.append("FEDERATED LEARNING EXPERIMENT SUMMARY")
    report.append("="*60)
    
    if 'final_accuracy' in results:
        report.append(f"Final Test Accuracy: {results['final_accuracy']:.2f}%")
    
    if 'final_loss' in results:
        report.append(f"Final Test Loss: {results['final_loss']:.4f}")
    
    if 'total_carbon_emission' in results:
        report.append(f"Total Carbon Emission: {results['total_carbon_emission']:.6f} kg CO2")
    
    if 'total_training_time' in results:
        report.append(f"Total Training Time: {results['total_training_time']:.2f} seconds")
    
    history = results.get('training_history', {})
    if 'rounds' in history:
        report.append(f"Total Rounds: {len(history['rounds'])}")
    
    if 'privacy_spent' in results:
        report.append(f"Privacy Budget Spent: {results['privacy_spent']:.4f}")
    
    report.append("="*60)
    
    return "\n".join(report)