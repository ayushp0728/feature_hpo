"""Result analysis utilities for HPO."""

from typing import Any, Callable, Optional

from .config.hpo_config import TrialResult


class ResultAnalyzer:
    """Analyzer for hyperparameter optimization trial results.

    This class provides utilities for extracting metrics from trial results
    and identifying the best performing configurations.

    Attributes:
        metric_name: Name of the metric to extract from results.
        metric_mode: 'max' to maximize metric, 'min' to minimize.
        metric_extractor: Custom function to extract metric from result objects.
    """

    def __init__(
        self,
        metric_name: str = "reward",
        metric_mode: str = "max",
        metric_extractor: Optional[Callable[[Any], float]] = None,
    ):
        """Initialize result analyzer.

        Args:
            metric_name: Name of the metric to optimize.
            metric_mode: 'max' to maximize metric, 'min' to minimize metric.
            metric_extractor: Optional custom function to extract metric value
                from result objects. If None, tries common attribute access patterns.
        """
        if metric_mode not in ["max", "min"]:
            raise ValueError("metric_mode must be 'max' or 'min'")

        self.metric_name = metric_name
        self.metric_mode = metric_mode
        self.metric_extractor = metric_extractor or self._default_metric_extractor

    def _default_metric_extractor(self, result: Any) -> float:
        """Default method to extract metric from result object.

        Tries several common patterns:
        1. Direct attribute access (result.reward)
        2. Dictionary access (result['reward'])
        3. Nested metric dict (result.metrics['reward'])

        Args:
            result: Result object from a learner.

        Returns:
            Extracted metric value.

        Raises:
            ValueError: If metric cannot be extracted from result.
        """
        # Try direct attribute access
        if hasattr(result, self.metric_name):
            value = getattr(result, self.metric_name)
            return float(value)

        # Try dictionary access
        if isinstance(result, dict) and self.metric_name in result:
            return float(result[self.metric_name])

        # Try nested metrics dict
        if hasattr(result, "metrics") and isinstance(result.metrics, dict):
            if self.metric_name in result.metrics:
                return float(result.metrics[self.metric_name])

        # Try metric_values_per_iteration (common in ROSE learners)
        if hasattr(result, "metric_values_per_iteration"):
            metrics = result.metric_values_per_iteration
            if isinstance(metrics, dict) and self.metric_name in metrics:
                # Get last value if it's a list
                values = metrics[self.metric_name]
                if isinstance(values, list) and values:
                    return float(values[-1])
                return float(values)

        raise ValueError(
            f"Could not extract metric '{self.metric_name}' from result. "
            f"Please provide a custom metric_extractor function."
        )

    def extract_metric(self, result: Any) -> float:
        """Extract metric value from a result object.

        Args:
            result: Result object from a learner.

        Returns:
            Extracted metric value as a float.
        """
        return self.metric_extractor(result)

    def find_best_trial(self, trials: list[TrialResult]) -> TrialResult:
        """Find the best performing trial.

        Args:
            trials: List of trial results to analyze.

        Returns:
            The trial with the best metric value.

        Raises:
            ValueError: If trials list is empty.
        """
        if not trials:
            raise ValueError("Cannot find best trial from empty list")

        if self.metric_mode == "max":
            return max(trials, key=lambda t: t.metric_value)
        else:
            return min(trials, key=lambda t: t.metric_value)

    def get_top_k_trials(self, trials: list[TrialResult], k: int = 5) -> list[TrialResult]:
        """Get top k performing trials.

        Args:
            trials: List of trial results to analyze.
            k: Number of top trials to return.

        Returns:
            List of top k trials sorted by performance.
        """
        reverse = self.metric_mode == "max"
        sorted_trials = sorted(trials, key=lambda t: t.metric_value, reverse=reverse)
        return sorted_trials[:k]

    def compute_statistics(self, trials: list[TrialResult]) -> dict[str, float]:
        """Compute statistics over all trials.

        Args:
            trials: List of trial results to analyze.

        Returns:
            Dictionary containing mean, std, min, max of metric values.
        """
        if not trials:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        values = [t.metric_value for t in trials]
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_val = variance**0.5

        return {
            "mean": mean_val,
            "std": std_val,
            "min": min(values),
            "max": max(values),
        }