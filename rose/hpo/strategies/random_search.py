# rose/hpo/strategies/random_search.py

from typing import Dict, List, Tuple
from .base import BaseStrategy


class RandomSearch(BaseStrategy):
    def __init__(self, search_space):
        self.space = search_space
        self.history: List[Tuple[Dict, float]] = []

    def propose(self, n: int) -> List[Dict]:
        return [self.space.sample() for _ in range(n)]

    def update(self, configs: List[Dict], metrics: List[float]) -> None:
        for cfg, metric in zip(configs, metrics):
            self.history.append((cfg, metric))

    def best(self) -> Dict:
        if not self.history:
            return {}

        best_cfg, best_metric = min(
            self.history, key=lambda x: x[1]
        )
        return {
            "config": best_cfg,
            "metric": best_metric,
        }