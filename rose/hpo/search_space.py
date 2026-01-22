"""
File to describe tunable hyperparameters for ROSE.

Allows me to validate different inputs for research. 
"""
class SearchSpace: 

    def __init__(self, space):
        self.space = space
        self.validate_space()

    def validate_space(self):
        for k,v in self.space.items():
            if not isinstance(v, list) or len(v) == 0:
                raise ValueError(f"Hyperparameter {k} must be a non-empty list of values.")
    
    def items(self):
        return self.space.items()