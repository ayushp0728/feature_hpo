# rose/hpo/controller.py

from typing import Any, Dict, List

from rose.al import ParallelActiveLearner
from rose import LearnerConfig, TaskConfig


class ALHPOController:
    """
    Active-Learning-specific HPO controller.

    Semantics:
    - One HPO iteration == one ParallelActiveLearner execution
    - Each parallel learner == one hyperparameter configuration
    """

    def __init__(
        self,
        learner: ParallelActiveLearner,
        strategy: Any,
        max_iter: int,
        metric_name: str = "loss",
    ):
        if not isinstance(learner, ParallelActiveLearner):
            raise TypeError(
                "ALHPOController requires ParallelActiveLearner "
                "(RL support is future work)."
            )

        self.learner = learner
        self.strategy = strategy
        self.max_iter = max_iter
        self.metric_name = metric_name

        self._validate_strategy()

    def _validate_strategy(self) -> None:
        for method in ("propose", "update", "best"):
            if not hasattr(self.strategy, method):
                raise TypeError(
                    f"HPO strategy must implement `{method}`"
                )

    async def run(
        self,
        iterations: int,
        parallelism: int,
    ) -> Dict[str, Any]:
        """
        Run HPO for a fixed number of iterations.

        Args:
            iterations: number of HPO rounds
            parallelism: number of configs per round (PAL learners)
        """

        for _ in range(iterations):

            # 1. Ask strategy for configs
            configs: List[Dict[str, Any]] = self.strategy.propose(parallelism)

            # 2. Convert configs → LearnerConfigs
            learner_configs = [
                LearnerConfig(
                    training=TaskConfig(kwargs=cfg)
                )
                for cfg in configs
            ]

            # 3. Run PAL ONCE
            await self.learner.teach(
                parallel_learners=parallelism,
                max_iter=self.max_iter,
                learner_configs=learner_configs,
            )

            # 4. Collect final metrics
            metrics = self._collect_metrics(parallelism)

            # 5. Update strategy
            self.strategy.update(configs, metrics)

        return self.strategy.best()

    def _collect_metrics(self, parallelism: int) -> List[float]:
        """
        Extract final metric per PAL learner.
        """
        values = []

        for i in range(parallelism):
            key = f"learner-{i}"
            history = self.learner.metric_values_per_iteration.get(key)

            if not history:
                values.append(float("inf"))
                continue

            last_iter = max(history.keys())
            values.append(history[last_iter])

        return values