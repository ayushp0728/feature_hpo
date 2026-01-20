import random
class RandomSearch(HPOStrategy):
    def __init__(self, search_space):
        self.space = search_space

    def suggest(self, history):
        return {
            name: random.choice(choices)
            for name, choices in self.space.items()
        }