
class HPOOptimizer:
    def __init__(
            self,
            strategy,
            learner_adapter,
            executor,
            max_evals,
            batch_size
    ):
        self.strategy = strategy
        self.learner_adapter = learner_adapter
        self.executor = executor
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.history = []

#to do: implement run method, which will use the strategy to suggest new hyperparameters, use the executor to evaluate them, and update the history accordingly