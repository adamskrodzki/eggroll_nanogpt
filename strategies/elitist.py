import torch

from .base import EGGROLLStrategy


class ElitistEGGROLL(EGGROLLStrategy):
    def __init__(self, alpha, sigma, rank, pop_size):
        super().__init__(alpha, sigma, rank, pop_size)
        self.prev_avg_loss = None
        self.prev_pop_A = None
        self.prev_pop_B = None

    def sample_population(self, linear_layers, device):
        if self.prev_pop_A is None:
            self.prev_pop_A = [None] * len(linear_layers)
            self.prev_pop_B = [None] * len(linear_layers)

        pop_size = self.pop_size
        rank = self.rank

        if self.generation == 0:
            for i, layer in enumerate(linear_layers):
                A = torch.randn(pop_size, layer.out_features, rank,
                                device=device, dtype=torch.float32)
                B = torch.randn(pop_size, layer.in_features, rank,
                                device=device, dtype=torch.float32)
                self.prev_pop_A[i] = A.clone()
                self.prev_pop_B[i] = B.clone()
                layer.set_population(A, B)
        else:
            sorted_idx = torch.argsort(self.prev_avg_loss)
            k = pop_size // 10
            keep_idx = sorted_idx[:k]

            for i, layer in enumerate(linear_layers):
                A_prev = self.prev_pop_A[i]
                B_prev = self.prev_pop_B[i]

                A_keep = A_prev[keep_idx]
                B_keep = B_prev[keep_idx]

                remaining = pop_size - k
                base = remaining // k
                extra = remaining - base * k

                counts = torch.full((k,), base, dtype=torch.long)
                if extra > 0:
                    counts[:extra] += 1

                parent_indices = torch.arange(k, device='cpu').repeat_interleave(counts.cpu()).to(device)
                A_parents = A_keep[parent_indices]
                B_parents = B_keep[parent_indices]

                children_A = 0.5 * A_parents + 0.5 * torch.randn_like(A_parents)
                children_B = 0.5 * B_parents + 0.5 * torch.randn_like(B_parents)

                new_A = torch.cat([A_keep, children_A], dim=0)
                new_B = torch.cat([B_keep, children_B], dim=0)

                self.prev_pop_A[i] = new_A.clone()
                self.prev_pop_B[i] = new_B.clone()
                layer.set_population(new_A, new_B)

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

    def on_generation_end(self, avg_loss):
        self.prev_avg_loss = avg_loss.detach().clone()
        super().on_generation_end(avg_loss)
