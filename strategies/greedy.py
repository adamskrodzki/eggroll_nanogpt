import torch

from .base import EGGROLLStrategy


class GreedyEGGROLL(EGGROLLStrategy):
    def sample_population(self, linear_layers, device):
        for layer in linear_layers:
            A = torch.randn(self.pop_size, layer.out_features, self.rank,
                            device=device, dtype=torch.float32)
            B = torch.randn(self.pop_size, layer.in_features, self.rank,
                            device=device, dtype=torch.float32)
            layer.set_population(A, B)

    def compute_update(self, linear_layers, fitness, avg_loss):
        alpha = self.alpha
        rank = self.rank
        best_idx = torch.argmax(fitness).item()
        best_fit = fitness[best_idx]
        for layer in linear_layers:
            if rank == 1:
                A_best = layer.A[best_idx].squeeze(-1)
                B_best = layer.B[best_idx].squeeze(-1)
                delta = best_fit * alpha * torch.outer(A_best, B_best)
            else:
                A_best = layer.A[best_idx]
                B_best = layer.B[best_idx]
                delta = best_fit * alpha * (A_best @ B_best.T)
            layer.M.data += delta
            layer.set_population(None, None)
