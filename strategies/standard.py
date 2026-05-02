import torch

from .base import EGGROLLStrategy


class StandardEGGROLL(EGGROLLStrategy):
    def sample_population(self, linear_layers, device):
        for layer in linear_layers:
            A = torch.randn(self.pop_size, layer.out_features, self.rank,
                            device=device, dtype=torch.float32)
            B = torch.randn(self.pop_size, layer.in_features, self.rank,
                            device=device, dtype=torch.float32)
            layer.set_population(A, B)

    def compute_update(self, linear_layers, fitness, avg_loss):
        N = self.pop_size
        alpha = self.alpha
        sigma = self.sigma
        for layer in linear_layers:
            # A: (N, out_features, rank), B: (N, in_features, rank)
            A = layer.A
            B = layer.B
            # delta: (out_features, in_features) = (alpha/(N*sigma)) * Σ_n fitness[n] * A[n] @ B[n].T
            # einsum: sum over n (pop) and r (rank), keep o (out) and i (in)
            delta = (alpha / (N * sigma)) * torch.einsum('n,nor,nir->oi', fitness, A, B)
            assert delta.shape == layer.M.shape, f"expected ({layer.out_features}, {layer.in_features}), got {delta.shape}"
            layer.M.data += delta
            layer.set_population(None, None)
