"""
Random client orchestrator for federated learning.
"""

import random
from typing import List, Any
import logging
from .base import BaseOrchestrator

logger = logging.getLogger(__name__)


class RandomOrchestrator(BaseOrchestrator):
    """
    Random client selection orchestrator.
    
    Selects clients randomly from the available pool.
    This is the simplest orchestration strategy.
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize random orchestrator.
        
        Args:
            seed: Random seed for reproducibility
        """
        super().__init__("RandomOrchestrator")
        if seed is not None:
            random.seed(seed)
        self.seed = seed
        
        logger.info(f"Initialized random orchestrator with seed {seed}")
    
    def select_clients(
        self,
        available_clients: List[Any],
        num_select: int,
        round_num: int,
        **kwargs
    ) -> List[Any]:
        """
        Randomly select clients for the current round.
        
        Args:
            available_clients: List of all available clients
            num_select: Number of clients to select
            round_num: Current round number
            **kwargs: Additional parameters (ignored)
        
        Returns:
            List of randomly selected clients
        """
        if num_select >= len(available_clients):
            # Select all clients if we need more than available
            selected = available_clients.copy()
        else:
            # Randomly sample clients
            selected = random.sample(available_clients, num_select)
        
        logger.info(f"Round {round_num}: Randomly selected {len(selected)} clients")
        
        return selected