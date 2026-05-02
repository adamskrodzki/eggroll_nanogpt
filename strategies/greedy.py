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
        sigma = self.sigma
        best_idx = torch.argmax(fitness).item()
        best_fit = fitness[best_idx]
        for layer in linear_layers:
            # A_best: (out_features, rank), B_best: (in_features, rank)
            A_best = layer.A[best_idx]
            B_best = layer.B[best_idx]
            # delta: (out_features, in_features) = best_fit * (alpha/sigma) * A_best @ B_best.T
            delta = best_fit * (alpha / sigma) * (A_best @ B_best.T)
            assert delta.shape == layer.M.shape, f"expected ({layer.out_features}, {layer.in_features}), got {delta.shape}"
            layer.M.data += delta
            layer.set_population(None, None)
