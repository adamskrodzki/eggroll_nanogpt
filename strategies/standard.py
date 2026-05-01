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
        rank = self.rank
        for layer in linear_layers:
            A = layer.A.squeeze(-1) if rank == 1 else layer.A.view(N, -1, rank)
            B = layer.B.squeeze(-1) if rank == 1 else layer.B.view(N, -1, rank)
            if rank == 1:
                delta = (alpha / N) * (A.T @ (fitness.unsqueeze(1) * B))
            else:
                delta = (alpha / N) * torch.einsum('nr,ni,no->oi', fitness, B, A)
            layer.M.data += delta
            layer.set_population(None, None)
