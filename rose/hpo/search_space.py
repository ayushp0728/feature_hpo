import random


class SearchSpace:
    """
    Defines hyperparameter ranges and sampling logic.
    """

    def __init__(self, space: dict):
        self.space = space

    def sample(self) -> dict:
        config = {}
        for name, spec in self.space.items():
            if spec["type"] == "uniform":
                config[name] = random.uniform(*spec["bounds"])
            elif spec["type"] == "choice":
                config[name] = random.choice(spec["values"])
            else:
                raise ValueError(f"Unknown spec type for {name}")
        return config