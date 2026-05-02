import torch

from .base import EGGROLLStrategy


class SequentialElitistEGGROLL(EGGROLLStrategy):
    def __init__(self, alpha, sigma, rank, pop_size, use_antithetic=True):
        super().__init__(alpha, sigma, rank, pop_size, use_antithetic)
        self.pop_A = None
        self.pop_B = None
        self.layer_loss = None

    def sample_population(self, linear_layers, device):
        if self.pop_A is None:
            self.pop_A = [None] * len(linear_layers)
            self.pop_B = [None] * len(linear_layers)
            self.layer_loss = [None] * len(linear_layers)

        half_size = self._gen_half_size()

        if self.generation == 0:
            for i, layer in enumerate(linear_layers):
                A, B = self._make_antithetic_linear(
                    torch.randn(half_size, layer.out_features, self.rank,
                                device=device, dtype=torch.float32),
                    torch.randn(half_size, layer.in_features, self.rank,
                                device=device, dtype=torch.float32)
                )
                self.pop_A[i] = A
                self.pop_B[i] = B
                bias_noise = None
                if layer.bias is not None:
                    bias_noise = self._make_antithetic_flat(
                        torch.randn(half_size, layer.out_features,
                                    device=device, dtype=torch.float32)
                    )
                layer.set_population(A, B, bias_noise)
        else:
            layer_idx = (self.generation - 1) % len(linear_layers)
            layer = linear_layers[layer_idx]

            prev_A = self.pop_A[layer_idx]
            prev_loss = self.layer_loss[layer_idx]

            if prev_A is None or prev_loss is None:
                A, B = self._make_antithetic_linear(
                    torch.randn(half_size, layer.out_features, self.rank,
                                device=device, dtype=torch.float32),
                    torch.randn(half_size, layer.in_features, self.rank,
                                device=device, dtype=torch.float32)
                )
            else:
                sorted_idx = torch.argsort(prev_loss)
                k = max(1, self.pop_size // 10)
                keep_idx = sorted_idx[:k]

                A_keep = prev_A[keep_idx]
                B_keep = self.pop_B[layer_idx][keep_idx]

                remaining = self.pop_size - k
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

                A = torch.cat([A_keep, children_A], dim=0)
                B = torch.cat([B_keep, children_B], dim=0)

            self.pop_A[layer_idx] = A
            self.pop_B[layer_idx] = B
            bias_noise = None
            if layer.bias is not None:
                bias_noise = self._make_antithetic_flat(
                    torch.randn(half_size, layer.out_features,
                                device=device, dtype=torch.float32)
                )
            layer.set_population(A, B, bias_noise)

        if hasattr(self, 'ln_layers') and hasattr(self, 'wpe_module'):
            self._sample_nonlinear_noise(self.ln_layers, self.wpe_module, device)

    def compute_update(self, linear_layers, fitness, avg_loss):
        N = self.pop_size
        alpha = self.alpha
        sigma = self.sigma

        if self.generation == 0:
            for i, layer in enumerate(linear_layers):
                A = layer.A
                B = layer.B
                delta = (alpha / (N * sigma)) * torch.einsum('n,nor,nir->oi', fitness, A, B)
                assert delta.shape == layer.M.shape
                layer.M.data += delta
                self._update_linear_bias(layer, fitness)
                layer.set_population(None, None)
            for i in range(len(linear_layers)):
                self.layer_loss[i] = avg_loss.detach().clone()
        else:
            layer_idx = (self.generation - 1) % len(linear_layers)
            layer = linear_layers[layer_idx]
            A = layer.A
            B = layer.B
            delta = (alpha / (N * sigma)) * torch.einsum('n,nor,nir->oi', fitness, A, B)
            assert delta.shape == layer.M.shape
            layer.M.data += delta
            self._update_linear_bias(layer, fitness)
            layer.set_population(None, None)
            self.layer_loss[layer_idx] = avg_loss.detach().clone()

        if hasattr(self, 'ln_layers') and hasattr(self, 'wpe_module'):
            self._update_nonlinear_params(self.ln_layers, self.wpe_module, fitness)

    def on_generation_end(self, avg_loss):
        super().on_generation_end(avg_loss)
