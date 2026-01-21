class LearnerAdapter: 
    """"
    Adapter to interface with different learners in HPO strategies.
    Allows for a consistent API regardless of the underlying learner implementation.
    """ 
    def __init__(self, learner_cls, metric_key): 
        self.learner_cls = learner_cls
        self.metric_key = metric_key

    def evaluate(self, params):
        """Evaluate the learner with given hyperparameters and return the metric."""
        learner = self.learner_cls(**params)
        metric = learner.train()
        return metric[self.metric_key]