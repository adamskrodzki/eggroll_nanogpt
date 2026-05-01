from abc import ABC, abstractmethod
from typing import List, Optional

import torch


class EGGROLLStrategy(ABC):
    def __init__(self, alpha: float, sigma: float, rank: int, pop_size: int):
        self.alpha = alpha
        self.sigma = sigma
        self.rank = rank
        self.pop_size = pop_size
        self.generation = 0
        self.best_loss_so_far = float('inf')
        self.last_loss: Optional[float] = None

    @abstractmethod
    def sample_population(self, linear_layers: List, device: torch.device) -> None:
        ...

    @abstractmethod
    def compute_update(self, linear_layers: List, fitness: torch.Tensor,
                       avg_loss: torch.Tensor) -> None:
        ...

    def on_generation_end(self, avg_loss: torch.Tensor) -> None:
        self.last_loss = avg_loss.mean().item()
        min_loss = avg_loss.min().item()
        if min_loss < self.best_loss_so_far:
            self.best_loss_so_far = min_loss
        self.generation += 1
