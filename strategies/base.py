from abc import ABC, abstractmethod
from typing import List, Optional

import torch


class EGGROLLStrategy(ABC):
    def __init__(self, alpha: float, sigma: float, rank: int, pop_size: int,
                 use_antithetic: bool = True):
        self.alpha = alpha
        self.sigma = sigma
        self.rank = rank
        self.pop_size = pop_size
        self.use_antithetic = use_antithetic
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

    def _gen_half_size(self):
        return self.pop_size // 2 if self.use_antithetic else self.pop_size

    def _make_antithetic_linear(self, A_in, B_in):
        if not self.use_antithetic:
            return A_in, B_in
        assert self.pop_size % 2 == 0, "pop_size must be even when use_antithetic=True"
        half = A_in.shape[0]
        A_full = torch.stack([A_in, -A_in], dim=1).reshape(self.pop_size, *A_in.shape[1:])
        B_full = torch.stack([B_in, B_in], dim=1).reshape(self.pop_size, *B_in.shape[1:])
        return A_full, B_full

    def _make_antithetic_flat(self, noise_in):
        if not self.use_antithetic:
            return noise_in
        assert self.pop_size % 2 == 0, "pop_size must be even when use_antithetic=True"
        full = torch.stack([noise_in, -noise_in], dim=1).reshape(self.pop_size, *noise_in.shape[1:])
        return full

    def _compute_fitness(self, avg_loss):
        fitness = -avg_loss.detach()
        return fitness - fitness.mean()

    def _sample_nonlinear_noise(self, ln_layers, wpe_module, device):
        half_size = self._gen_half_size()
        for ln in ln_layers:
            w_noise = self._make_antithetic_flat(
                torch.randn(half_size, ln.weight.shape[0], device=device, dtype=torch.float32)
            )
            b_noise = None
            if ln.bias is not None:
                b_noise = self._make_antithetic_flat(
                    torch.randn(half_size, ln.bias.shape[0], device=device, dtype=torch.float32)
                )
            ln.set_noise(w_noise, b_noise)
        if wpe_module is not None:
            wpe_noise = self._make_antithetic_flat(
                torch.randn(half_size, wpe_module.config.block_size,
                            wpe_module.config.n_embd, device=device, dtype=torch.float32)
            )
            wpe_module.set_wpe_noise(wpe_noise)

    def _update_nonlinear_params(self, ln_layers, wpe_module, fitness):
        N = self.pop_size
        alpha = self.alpha
        sigma = self.sigma
        for ln in ln_layers:
            scale = 1.0 / ln.weight.shape[0]
            if ln.weight_noise is not None:
                delta = scale * (alpha / (N * sigma)) * (fitness[:, None] * ln.weight_noise).sum(0)
                ln.weight.data += delta
            if ln.bias_noise is not None:
                delta = scale * (alpha / (N * sigma)) * (fitness[:, None] * ln.bias_noise).sum(0)
                ln.bias.data += delta
            ln.set_noise(None, None)
        if wpe_module is not None and wpe_module.wpe_noise is not None:
            scale = 1.0 / wpe_module.config.n_embd
            delta = scale * (alpha / (N * sigma)) * torch.einsum('n,ntc->tc', fitness, wpe_module.wpe_noise)
            wpe_module.transformer.wpe.weight.data += delta
            wpe_module.set_wpe_noise(None)

    def _update_nonlinear_params_greedy(self, ln_layers, wpe_module, fitness):
        alpha = self.alpha
        sigma = self.sigma
        best_idx = torch.argmax(fitness).item()
        best_fit = fitness[best_idx]
        for ln in ln_layers:
            scale = 1.0 / ln.weight.shape[0]
            if ln.weight_noise is not None:
                ln.weight.data += best_fit * (alpha / sigma) * scale * ln.weight_noise[best_idx]
            if ln.bias_noise is not None:
                ln.bias.data += best_fit * (alpha / sigma) * scale * ln.bias_noise[best_idx]
            ln.set_noise(None, None)
        if wpe_module is not None and wpe_module.wpe_noise is not None:
            scale = 1.0 / wpe_module.config.n_embd
            wpe_module.transformer.wpe.weight.data += best_fit * (alpha / sigma) * scale * wpe_module.wpe_noise[best_idx]
            wpe_module.set_wpe_noise(None)

    def _update_linear_bias(self, layer, fitness):
        if layer.bias is not None and layer.bias_noise is not None:
            N = self.pop_size
            alpha = self.alpha
            sigma = self.sigma
            scale = 1.0 / layer.out_features
            delta = scale * (alpha / (N * sigma)) * (fitness[:, None] * layer.bias_noise).sum(0)
            layer.bias.data += delta

    def _update_linear_bias_greedy(self, layer, fitness, best_idx, best_fit):
        if layer.bias is not None and layer.bias_noise is not None:
            alpha = self.alpha
            sigma = self.sigma
            scale = 1.0 / layer.out_features
            layer.bias.data += best_fit * (alpha / sigma) * scale * layer.bias_noise[best_idx]
