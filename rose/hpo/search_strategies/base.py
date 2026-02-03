"""Base class for hyperparameter search strategies."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class SearchStrategy(ABC):
    """Abstract base class for hyperparameter search strategies.

    Search strategies are responsible for generating hyperparameter
    configurations to try during optimization. Different strategies
    (grid search, random search, Bayesian optimization, etc.) implement
    different approaches to exploring the hyperparameter space.
    """

    @abstractmethod
    def suggest_configs(
        self,
        space: dict[str, list[Any]],
        n_configs: int,
        previous_results: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        """Suggest hyperparameter configurations to try.

        Args:
            space: Dictionary defining the search space.
                Keys are parameter names, values are lists of possible values.
            n_configs: Number of configurations to suggest.
            previous_results: Optional list of previous trial results.
                Each result contains 'hyperparameters' and 'metric_value'.
                Used by adaptive strategies (e.g., Bayesian optimization).

        Returns:
            List of hyperparameter configuration dictionaries.
            Each dictionary maps parameter names to values.

        Example:
            >>> strategy = GridSearch()
            >>> space = {'lr': [0.01, 0.1], 'batch_size': [32, 64]}
            >>> configs = strategy.suggest_configs(space, n_configs=2)
            >>> configs
            [{'lr': 0.01, 'batch_size': 32}, {'lr': 0.01, 'batch_size': 64}]
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the search strategy state.

        Called when starting a new optimization run to clear any
        internal state from previous runs.
        """
        pass

    def _validate_space(self, space: dict[str, list[Any]]) -> None:
        """Validate that the search space is properly formatted.

        Args:
            space: Search space to validate.

        Raises:
            ValueError: If space is empty or improperly formatted.
        """
        if not space:
            raise ValueError("Search space cannot be empty")

        for param_name, values in space.items():
            if not isinstance(values, list):
                raise ValueError(
                    f"Values for '{param_name}' must be a list, got {type(values)}"
                )
            if not values:
                raise ValueError(f"Values list for '{param_name}' cannot be empty")