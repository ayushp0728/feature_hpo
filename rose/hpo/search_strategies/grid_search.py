"""Grid search strategy for hyperparameter optimization."""

import itertools
from typing import Any, Optional

from .base import SearchStrategy


class GridSearch(SearchStrategy):
    """Grid search strategy that systematically explores all combinations.

    Grid search creates a Cartesian product of all hyperparameter values
    and systematically tries each combination. This ensures complete
    coverage of the search space but can be computationally expensive
    for large spaces.

    Attributes:
        _all_configs: Complete list of all possible configurations.
        _current_index: Current position in the configuration list.
    """

    def __init__(self):
        """Initialize grid search strategy."""
        self._all_configs: list[dict[str, Any]] = []
        self._current_index: int = 0

    def suggest_configs(
        self,
        space: dict[str, list[Any]],
        n_configs: int,
        previous_results: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        """Suggest next batch of configurations from the grid.

        Args:
            space: Dictionary defining the search space.
            n_configs: Number of configurations to suggest.
            previous_results: Ignored for grid search (not adaptive).

        Returns:
            List of n_configs hyperparameter configurations.
            If fewer than n_configs remain, returns all remaining configs.

        Example:
            >>> grid = GridSearch()
            >>> space = {'lr': [0.01, 0.1], 'bs': [32, 64]}
            >>> grid.suggest_configs(space, n_configs=2)
            [{'lr': 0.01, 'bs': 32}, {'lr': 0.01, 'bs': 64}]
        """
        self._validate_space(space)

        # Generate all configs if not already done
        if not self._all_configs:
            self._generate_grid(space)

        # Get next batch of configs
        start_idx = self._current_index
        end_idx = min(start_idx + n_configs, len(self._all_configs))

        configs = self._all_configs[start_idx:end_idx]
        self._current_index = end_idx

        return configs

    def _generate_grid(self, space: dict[str, list[Any]]) -> None:
        """Generate all possible configurations from the search space.

        Args:
            space: Dictionary defining the search space.
        """
        # Get parameter names and their possible values
        param_names = list(space.keys())
        param_values = [space[name] for name in param_names]

        # Create Cartesian product of all values
        all_combinations = itertools.product(*param_values)

        # Convert to list of dicts
        self._all_configs = [
            dict(zip(param_names, combination)) for combination in all_combinations
        ]

    def reset(self) -> None:
        """Reset grid search to start from beginning."""
        self._current_index = 0
        self._all_configs = []

    def get_total_configs(self) -> int:
        """Get total number of configurations in the grid.

        Returns:
            Total number of configurations, or 0 if grid not generated yet.
        """
        return len(self._all_configs)

    def is_exhausted(self) -> bool:
        """Check if all configurations have been suggested.

        Returns:
            True if all configurations have been suggested, False otherwise.
        """
        return self._current_index >= len(self._all_configs)