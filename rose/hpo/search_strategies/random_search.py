"""Random search strategy for hyperparameter optimization."""

import random
from typing import Any, Optional

from .base import SearchStrategy


class RandomSearch(SearchStrategy):
    """Random search strategy that samples configurations randomly.

    Random search randomly samples hyperparameter configurations from
    the search space. This is often more efficient than grid search
    for high-dimensional spaces, as it doesn't waste time on unlikely
    combinations.

    Attributes:
        seed: Random seed for reproducibility (optional).
        _rng: Random number generator instance.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize random search strategy.

        Args:
            seed: Random seed for reproducibility. If None, uses random seed.
        """
        self.seed = seed
        self._rng = random.Random(seed)

    def suggest_configs(
        self,
        space: dict[str, list[Any]],
        n_configs: int,
        previous_results: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        """Suggest random configurations from the search space.

        Args:
            space: Dictionary defining the search space.
            n_configs: Number of configurations to suggest.
            previous_results: Ignored for random search (not adaptive).

        Returns:
            List of n_configs randomly sampled hyperparameter configurations.

        Example:
            >>> random_search = RandomSearch(seed=42)
            >>> space = {'lr': [0.01, 0.1], 'bs': [32, 64]}
            >>> configs = random_search.suggest_configs(space, n_configs=2)
            >>> len(configs)
            2
        """
        self._validate_space(space)

        configs = []
        for _ in range(n_configs):
            config = {}
            for param_name, values in space.items():
                config[param_name] = self._rng.choice(values)
            configs.append(config)

        return configs

    def reset(self) -> None:
        """Reset random search with the same or new seed."""
        self._rng = random.Random(self.seed)

    def set_seed(self, seed: int) -> None:
        """Set a new random seed.

        Args:
            seed: New random seed.
        """
        self.seed = seed
        self._rng = random.Random(seed)