from typing import Any, Dict, List, Tuple


class HPOStrategy:
    """
    Base class for HPO strategies.

    Provides shared bookkeeping and a minimal contract.
    Concrete strategies decide *how* to propose configurations.
    """

    def __init__(self, minimize: bool = True):
        # Whether lower metric values are better
        self.minimize = minimize

        # History: list of (config, metric)
        self.history: List[Tuple[Dict[str, Any], float]] = []

        # Best-so-far tracking
        self.best_config: Dict[str, Any] | None = None
        self.best_metric: float | None = None

    # ------------------------------------------------------------------
    # Interface methods (must be implemented by subclasses)
    # ------------------------------------------------------------------

    def propose(self, n: int) -> List[Dict[str, Any]]:
        """
        Propose `n` hyperparameter configurations.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared logic (this is the important part)
    # ------------------------------------------------------------------

    def update(
        self,
        configs: List[Dict[str, Any]],
        metrics: List[float],
    ) -> None:
        """
        Update strategy state with observed results.
        """
        for cfg, metric in zip(configs, metrics):
            self.history.append((cfg, metric))
            self._update_best(cfg, metric)

    def _update_best(self, cfg: Dict[str, Any], metric: float) -> None:
        if self.best_metric is None:
            self.best_metric = metric
            self.best_config = cfg
            return

        is_better = (
            metric < self.best_metric
            if self.minimize
            else metric > self.best_metric
        )

        if is_better:
            self.best_metric = metric
            self.best_config = cfg

    def best(self) -> Dict[str, Any]:
        """
        Return the best configuration seen so far.
        """
        if self.best_config is None:
            raise RuntimeError("No configurations evaluated yet.")
        return self.best_config

    # ------------------------------------------------------------------
    # Utility helpers (optional but useful)
    # ------------------------------------------------------------------

    def num_evaluations(self) -> int:
        return len(self.history)

    def seen_configs(self) -> List[Dict[str, Any]]:
        return [cfg for cfg, _ in self.history]