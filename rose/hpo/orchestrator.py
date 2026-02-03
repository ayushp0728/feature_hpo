"""HPO Orchestrator - coordinates hyperparameter optimization."""

import time
from pathlib import Path
from typing import Any, Callable, Optional, Union

from ..learner import LearnerConfig
from .config.hpo_config import HPOConfig, HPOResult, TrialResult
from .result_analyzer import ResultAnalyzer
from .search_strategies.base import SearchStrategy


class HPOOrchestrator:
    """Orchestrates hyperparameter optimization for ROSE learners.

    The orchestrator works with any ParallelLearner (RL or Active Learning)
    to systematically explore hyperparameter spaces. It generates configurations
    using a search strategy, runs parallel trials, and identifies the best
    performing hyperparameters.

    Attributes:
        parallel_learner: ROSE parallel learner instance (ParallelReinforcementLearner
            or ParallelActiveLearner).
        search_strategy: Strategy for generating hyperparameter configurations.
        config: HPO configuration settings.
        result_analyzer: Analyzer for extracting and comparing trial results.
        trial_history: List of all completed trials.
    """

    def __init__(
        self,
        parallel_learner: Any,
        search_strategy: SearchStrategy,
        config: HPOConfig,
        metric_extractor: Optional[Callable[[Any], float]] = None,
    ):
        """Initialize HPO orchestrator.

        Args:
            parallel_learner: ROSE parallel learner instance. Currently only
                ParallelActiveLearner is supported (must have .teach() method).
            search_strategy: Strategy for exploring hyperparameter space.
            config: HPO configuration settings.
            metric_extractor: Optional custom function to extract metric values
                from learner results. If None, uses default extraction logic.

        Raises:
            ValueError: If parallel_learner doesn't have .teach() method.
        """
        # Validate parallel learner has required method
        if not hasattr(parallel_learner, "teach"):
            raise ValueError(
                "parallel_learner must have .teach() method. "
                "Currently only ParallelActiveLearner is supported."
            )

        self.parallel_learner = parallel_learner
        self.search_strategy = search_strategy
        self.config = config
        self.result_analyzer = ResultAnalyzer(
            metric_name=config.metric_name,
            metric_mode=config.metric_mode,
            metric_extractor=metric_extractor,
        )
        self.trial_history: list[TrialResult] = []
        self._trial_counter = 0

    async def optimize(self) -> HPOResult:
        """Run hyperparameter optimization.

        Executes multiple rounds of parallel trials, each round generating
        and testing different hyperparameter configurations. Returns the
        best configuration found across all trials.

        Returns:
            HPOResult containing best configuration and all trial results.

        Example:
            >>> orchestrator = HPOOrchestrator(
            ...     parallel_learner=my_rl_learner,
            ...     search_strategy=GridSearch(),
            ...     config=HPOConfig(
            ...         hyperparameter_space={'lr': [0.01, 0.1], 'bs': [32, 64]},
            ...         trials_per_round=4,
            ...         max_rounds=2
            ...     )
            ... )
            >>> result = await orchestrator.optimize()
            >>> print(result.best_config)
        """
        print(f"Starting HPO with {self.config.trials_per_round} trials per round")
        print(f"Search space: {self.config.hyperparameter_space}")

        start_time = time.time()
        self.search_strategy.reset()

        for round_num in range(self.config.max_rounds):
            print(f"\n{'=' * 60}")
            print(f"HPO Round {round_num + 1}/{self.config.max_rounds}")
            print(f"{'=' * 60}")

            # Generate hyperparameter configurations for this round
            hp_configs = self.search_strategy.suggest_configs(
                space=self.config.hyperparameter_space,
                n_configs=self.config.trials_per_round,
                previous_results=self._format_results_for_strategy(),
            )

            if not hp_configs:
                print("No more configurations to try. Stopping early.")
                break

            print(f"Testing {len(hp_configs)} configurations...")

            # Convert to LearnerConfig objects
            # First, prepare trial directories
            for i, hp in enumerate(hp_configs):
                trial_dir = Path(f"trial_{self._trial_counter + i}")
                trial_dir.mkdir(exist_ok=True)
                
                # Copy initial data files if they exist
                # Support multiple data file names (al_data.pkl, nn_data.pkl, etc.)
                import shutil
                for data_file in ["al_data.pkl", "nn_data.pkl", "data.pkl"]:
                    if Path(data_file).exists():
                        shutil.copy(data_file, trial_dir / data_file)
            
            learner_configs = [
                self._hyperparams_to_learner_config(hp, self._trial_counter + i) 
                for i, hp in enumerate(hp_configs)
            ]

            # Run parallel trials
            trial_results = await self._run_parallel_trials(
                learner_configs, round_num
            )

            # Process and store results
            for hp_config, result in zip(hp_configs, trial_results):
                metric_value = self.result_analyzer.extract_metric(result)

                trial = TrialResult(
                    trial_id=self._trial_counter,
                    hyperparameters=hp_config,
                    metric_value=metric_value,
                    full_result=result,
                    round_num=round_num,
                )
                self.trial_history.append(trial)
                self._trial_counter += 1

                print(
                    f"  Trial {trial.trial_id}: {hp_config} -> "
                    f"{self.config.metric_name}={metric_value:.4f}"
                )

            # Report best so far
            best_trial = self.result_analyzer.find_best_trial(self.trial_history)
            print(f"\nBest so far:")
            print(f"  Config: {best_trial.hyperparameters}")
            print(f"  {self.config.metric_name}: {best_trial.metric_value:.4f}")

        end_time = time.time()
        optimization_time = end_time - start_time

        # Generate final results
        best_trial = self.result_analyzer.find_best_trial(self.trial_history)

        result = HPOResult(
            best_config=best_trial.hyperparameters,
            best_metric=best_trial.metric_value,
            all_trials=self.trial_history,
            total_trials=len(self.trial_history),
            optimization_time=optimization_time,
        )

        print(f"\n{'=' * 60}")
        print("HPO Complete!")
        print(f"{'=' * 60}")
        print(result.summary())

        return result

    async def _run_parallel_trials(
        self, learner_configs: list[LearnerConfig], round_num: int
    ) -> list[Any]:
        """Run parallel trials using the parallel learner.

        Args:
            learner_configs: List of LearnerConfig objects for each trial.
            round_num: Current optimization round number.

        Returns:
            List of results from each trial.
        """
        # Call .teach() method for Active Learning
        results = await self.parallel_learner.teach(
            parallel_learners=len(learner_configs),
            max_iter=self.config.max_iter_per_trial,
            learner_configs=learner_configs,
            **self.config.learner_kwargs,
        )

        return results

    def _hyperparams_to_learner_config(
        self, hyperparams: dict[str, Any], trial_id: int
    ) -> LearnerConfig:
        """Convert hyperparameter dict to LearnerConfig.

        This method determines which LearnerConfig fields to populate based
        on the learner type. By default, hyperparameters are placed in the
        'training' field for Active Learning learners.

        Args:
            hyperparams: Dictionary of hyperparameter names and values.
                For Active Learning, these typically include training hyperparameters
                like learning_rate, batch_size, etc.
            trial_id: Unique ID for this trial, used to create isolated work directory.

        Returns:
            LearnerConfig object with hyperparameters assigned to appropriate fields.

        Note:
            Currently supports Active Learning learners only (those with .teach() method).
            Override this method in a subclass for custom mapping logic or to support
            other learner types.
        """
        from ..learner import TaskConfig
        
        # Add unique work directory for this trial to avoid file conflicts
        hyperparams_with_workdir = hyperparams.copy()
        hyperparams_with_workdir["--work_dir"] = f"trial_{trial_id}"
        
        # Wrap hyperparameters in TaskConfig
        # TaskConfig(kwargs={...}) will pass these as **kwargs to task functions
        task_config = TaskConfig(kwargs=hyperparams_with_workdir)
        
        # Detect learner type and assign to appropriate config field
        if hasattr(self.parallel_learner, "teach"):
            # ActiveLearner - put hyperparams in training config
            # This passes hyperparams to the training task function
            return LearnerConfig(training=task_config)
        elif hasattr(self.parallel_learner, "learn"):
            # ReinforcementLearner - TODO: implement proper field mapping
            raise NotImplementedError(
                "HPO for ReinforcementLearner not yet implemented. "
                "Currently only ParallelActiveLearner is supported."
            )
        else:
            # Unknown learner type
            raise ValueError(
                f"Unknown learner type. Learner must have .teach() method for Active Learning."
            )

    def _format_results_for_strategy(self) -> list[dict[str, Any]]:
        """Format trial history for search strategy consumption.

        Returns:
            List of dicts containing hyperparameters and metric values.
        """
        return [
            {"hyperparameters": trial.hyperparameters, "metric_value": trial.metric_value}
            for trial in self.trial_history
        ]

    def get_statistics(self) -> dict[str, float]:
        """Get statistics over all completed trials.

        Returns:
            Dictionary with mean, std, min, max of metric values.
        """
        return self.result_analyzer.compute_statistics(self.trial_history)

    def get_top_k_configs(self, k: int = 5) -> list[dict[str, Any]]:
        """Get top k performing configurations.

        Args:
            k: Number of top configurations to return.

        Returns:
            List of hyperparameter configurations sorted by performance.
        """
        top_trials = self.result_analyzer.get_top_k_trials(self.trial_history, k)
        return [trial.hyperparameters for trial in top_trials]