"""HPO (Hyperparameter Optimization) module for ROSE.

This module provides intelligent hyperparameter search capabilities for ROSE learners.
It works with any ParallelLearner (RL or Active Learning) to systematically explore
hyperparameter spaces and identify optimal configurations.
"""

from .config.hpo_config import HPOConfig, HPOResult
from .orchestrator import HPOOrchestrator
from .result_analyzer import ResultAnalyzer
from .search_strategies.base import SearchStrategy
from .search_strategies.grid_search import GridSearch
from .search_strategies.random_search import RandomSearch

__all__ = [
    "HPOOrchestrator",
    "HPOConfig",
    "HPOResult",
    "SearchStrategy",
    "GridSearch",
    "RandomSearch",
    "ResultAnalyzer",
]