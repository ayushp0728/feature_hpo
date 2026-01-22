class HPOController:
    """
    This class is responsible for managing the Hyperparameter Optimization (HPO).

    This will accept a rose learner, HPO strategy, Asyncflow engine, and search space. 

    This will implement optimization loop, batching, failure handling, etc.
    
    Coordinate learners, how many learners, which hyperparameters to try next, etc.
    """

    def __init__(self, learner, strategy, asyncflow):
        self.learner = learner
        self.strategy = strategy
        self.asyncflow = asyncflow
    
    async def run(
            self,
            iterations: int,
            parallelism: int,
            stop_on_failure: bool = True
    ):
        """
        Run the HPO process for a given number of iterations and parallelism level.

        Args:
            iterations (int): Number of hyperparameter configurations to try.
            parallelism (int): Number of concurrent learners to run.
            stop_on_failure (bool): Whether to stop on first failure or continue.
        """
        for i in range(iterations):
            tasks = []
            for _ in range(parallelism):
                try:
                    hyperparams = self.strategy.suggest(self.learner.get_history())
                    task = self.asyncflow.create_task(
                        self.learner.teach(
                            learner_names=[f"learner_{i}"],
                            learner_configs={f"learner_{i}": hyperparams},
                            model_names=["model"],
                            max_iter=1,
                        )
                    )
                    tasks.append(task)
                except Exception as e:
                    if stop_on_failure:
                        raise e
                    else:
                        print(f"Failed to suggest hyperparameters: {e}")
            if tasks:
                await self.asyncflow.gather(*tasks)
    
