"""
Base Orchestration Module.

This module provides the base class and interfaces for client orchestration
in federated learning systems.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseOrchestrator(ABC):
    """
    Abstract base class for client orchestration strategies.
    
    Orchestrators are responsible for selecting which clients participate
    in each round of federated learning based on various criteria.
    """
    
    def __init__(self, name: str = "BaseOrchestrator"):
        """
        Initialize the orchestrator.
        
        Args:
            name: Name of the orchestrator
        """
        self.name = name
        self.history = []
        logger.info(f"Initialized {self.name}")
    
    @abstractmethod
    def select_clients(
        self,
        available_clients: List[Any],
        num_select: int,
        round_num: int,
        **kwargs
    ) -> List[Any]:
        """
        Select clients for the current round.
        
        Args:
            available_clients: List of all available clients
            num_select: Number of clients to select
            round_num: Current round number
            **kwargs: Additional parameters specific to orchestration strategy
        
        Returns:
            List of selected clients
        """
        pass
    
    def update_history(self, round_results: dict) -> None:
        """
        Update orchestrator with results from the completed round.
        
        Args:
            round_results: Dictionary containing round results and metrics
        """
        self.history.append(round_results)
        logger.debug(f"{self.name} updated with round {round_results.get('round', 'unknown')} results")
    
    def get_selection_metrics(self) -> dict:
        """
        Get metrics about client selection patterns.
        
        Returns:
            Dictionary containing selection metrics
        """
        if not self.history:
            return {}
        
        # Count client participation
        client_participation = {}
        total_rounds = len(self.history)
        
        for round_result in self.history:
            selected_clients = round_result.get('selected_clients', [])
            for client_id in selected_clients:
                client_participation[client_id] = client_participation.get(client_id, 0) + 1
        
        # Calculate participation rates
        participation_rates = {
            client_id: count / total_rounds 
            for client_id, count in client_participation.items()
        }
        
        return {
            'total_rounds': total_rounds,
            'unique_clients_selected': len(client_participation),
            'client_participation_counts': client_participation,
            'client_participation_rates': participation_rates,
            'avg_clients_per_round': sum(len(r.get('selected_clients', [])) for r in self.history) / total_rounds
        }