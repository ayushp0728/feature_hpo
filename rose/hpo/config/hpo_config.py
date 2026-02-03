"""Configuration classes for HPO module."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class HPOConfig:
    """Configuration for hyperparameter optimization.

    Attributes:
        hyperparameter_space: Dictionary defining the search space.
            Keys are parameter names, values are lists of possible values.
            Example: {'learning_rate': [0.001, 0.01, 0.1], 'batch_size': [32, 64]}
        trials_per_round: Number of parallel trials to run per round.
        max_rounds: Maximum number of optimization rounds.
        max_iter_per_trial: Maximum iterations for each individual trial.
        metric_name: Name of the metric to optimize (e.g., 'reward', 'loss').
        metric_mode: 'max' to maximize metric, 'min' to minimize metric.
        learner_kwargs: Additional kwargs to pass to the learner's learn/teach method.
    """

    hyperparameter_space: dict[str, list[Any]]
    trials_per_round: int = 4
    max_rounds: int = 1
    max_iter_per_trial: int = 100
    metric_name: str = "reward"
    metric_mode: str = "max"
    learner_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if self.trials_per_round < 1:
            raise ValueError("trials_per_round must be >= 1")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if self.metric_mode not in ["max", "min"]:
            raise ValueError("metric_mode must be 'max' or 'min'")
        if not self.hyperparameter_space:
            raise ValueError("hyperparameter_space cannot be empty")


@dataclass
class TrialResult:
    """Result from a single hyperparameter trial.

    Attributes:
        trial_id: Unique identifier for this trial.
        hyperparameters: Hyperparameter configuration used in this trial.
        metric_value: Performance metric value achieved.
        full_result: Complete result object from the learner.
        round_num: Which optimization round this trial was part of.
        metadata: Additional information about the trial.
    """

    trial_id: int
    hyperparameters: dict[str, Any]
    metric_value: float
    full_result: Any
    round_num: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HPOResult:
    """Result from hyperparameter optimization.

    Attributes:
        best_config: Hyperparameter configuration that achieved best performance.
        best_metric: Best metric value achieved.
        all_trials: List of all trial results.
        total_trials: Total number of trials run.
        optimization_time: Total time spent on optimization (optional).
    """

    best_config: dict[str, Any]
    best_metric: float
    all_trials: list[TrialResult]
    total_trials: int
    optimization_time: Optional[float] = None

    def get_top_k_configs(self, k: int = 5) -> list[dict[str, Any]]:
        """Get top k performing configurations.

        Args:
            k: Number of top configurations to return.

        Returns:
            List of hyperparameter configurations sorted by performance.
        """
        sorted_trials = sorted(
            self.all_trials, key=lambda x: x.metric_value, reverse=True
        )
        return [trial.hyperparameters for trial in sorted_trials[:k]]

    def summary(self) -> str:
        """Generate a summary string of the optimization results."""
        summary = f"HPO Results Summary:\n"
        summary += f"  Total Trials: {self.total_trials}\n"
        summary += f"  Best Metric: {self.best_metric:.4f}\n"
        summary += f"  Best Config: {self.best_config}\n"
        if self.optimization_time:
            summary += f"  Time: {self.optimization_time:.2f}s\n"
        return summary