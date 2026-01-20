from itertools import product

class GridSearch(HPOStrategy):
    def __init__(self, search_space):
        self.space = search_space
        self.grid = self._create_grid(search_space)
        self.index = 0

    def _create_grid(self, search_space):
        
        keys = list(search_space.keys())
        values = [search_space[key] for key in keys]
        grid_points = list(product(*values))
        return [dict(zip(keys, point)) for point in grid_points]

    def suggest(self, history):
        if self.index >= len(self.grid):
            raise StopIteration("All grid points have been evaluated.")
        suggestion = self.grid[self.index]
        self.index += 1
        return suggestion