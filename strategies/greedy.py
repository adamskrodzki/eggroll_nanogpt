import torch

from .base import EGGROLLStrategy


class GreedyEGGROLL(EGGROLLStrategy):
    def sample_population(self, linear_layers, device):
        half_size = self._gen_half_size()
        for layer in linear_layers:
            A = self._make_antithetic_linear(
                torch.randn(half_size, layer.out_features, self.rank,
                            device=device, dtype=torch.float32),
                torch.randn(half_size, layer.in_features, self.rank,
                            device=device, dtype=torch.float32)
            )
            bias_noise = None
            if layer.bias is not None:
                bias_noise = self._make_antithetic_flat(
                    torch.randn(half_size, layer.out_features,
                                device=device, dtype=torch.float32)
                )
            layer.set_population(*A, bias_noise)
        if hasattr(self, 'ln_layers') and hasattr(self, 'wpe_module'):
            self._sample_nonlinear_noise(self.ln_layers, self.wpe_module, device)

    def compute_update(self, linear_layers, fitness, avg_loss):
        alpha = self.alpha
        sigma = self.sigma
        best_idx = torch.argmax(fitness).item()
        best_fit = fitness[best_idx]
        for layer in linear_layers:
            A_best = layer.A[best_idx]
            B_best = layer.B[best_idx]
            delta = best_fit * (alpha / sigma) * (A_best @ B_best.T)
            assert delta.shape == layer.M.shape, f"expected ({layer.out_features}, {layer.in_features}), got {delta.shape}"
            layer.M.data += delta
            self._update_linear_bias_greedy(layer, fitness, best_idx, best_fit)
            layer.set_population(None, None)
        if hasattr(self, 'ln_layers') and hasattr(self, 'wpe_module'):
            self._update_nonlinear_params_greedy(self.ln_layers, self.wpe_module, fitness)
