import torch

from .base import EGGROLLStrategy


class LayerGroupedEGGROLL(EGGROLLStrategy):
    def __init__(self, alpha, sigma, rank, pop_size, use_antithetic=True):
        super().__init__(alpha, sigma, rank, pop_size, use_antithetic)

    def _build_groups(self, linear_layers, ln_layers):
        n_blocks = len(ln_layers) // 2
        groups = []
        n_lin_per_block = len(linear_layers) // n_blocks
        for b in range(n_blocks):
            lin_start = b * n_lin_per_block
            lin_end = (b + 1) * n_lin_per_block
            ln_start = b * 2
            ln_end = (b + 1) * 2
            groups.append((linear_layers[lin_start:lin_end], ln_layers[ln_start:ln_end]))
        return groups

    def sample_population(self, linear_layers, device):
        if not hasattr(self, 'ln_layers'):
            raise RuntimeError("LayerGroupedEGGROLL requires ln_layers to be set")
        ln_layers = self.ln_layers
        groups = self._build_groups(linear_layers, ln_layers)
        half_size = self._gen_half_size()

        if self.generation == 0:
            for layer in linear_layers:
                A, B = self._make_antithetic_linear(
                    torch.randn(half_size, layer.out_features, self.rank,
                                device=device, dtype=torch.float32),
                    torch.randn(half_size, layer.in_features, self.rank,
                                device=device, dtype=torch.float32)
                )
                layer.set_population(A, B)
            for ln in ln_layers:
                w_noise = self._make_antithetic_flat(
                    torch.randn(half_size, ln.weight.shape[0], device=device, dtype=torch.float32)
                )
                if ln.bias is not None:
                    b_noise = self._make_antithetic_flat(
                        torch.randn(half_size, ln.bias.shape[0], device=device, dtype=torch.float32)
                    )
                else:
                    b_noise = None
                ln.set_noise(w_noise, b_noise)
            if hasattr(self, 'wpe_module') and self.wpe_module is not None:
                wpe_noise = self._make_antithetic_flat(
                    torch.randn(half_size, self.wpe_module.config.block_size,
                                self.wpe_module.config.n_embd, device=device, dtype=torch.float32)
                )
                self.wpe_module.set_wpe_noise(wpe_noise)
        else:
            group_idx = (self.generation - 1) % len(groups)
            active_lins, active_lns = groups[group_idx]

            for layer in linear_layers:
                layer.set_population(None, None)
            for ln in ln_layers:
                ln.set_noise(None, None)

            for layer in active_lins:
                A, B = self._make_antithetic_linear(
                    torch.randn(half_size, layer.out_features, self.rank,
                                device=device, dtype=torch.float32),
                    torch.randn(half_size, layer.in_features, self.rank,
                                device=device, dtype=torch.float32)
                )
                layer.set_population(A, B)
            for ln in active_lns:
                w_noise = self._make_antithetic_flat(
                    torch.randn(half_size, ln.weight.shape[0], device=device, dtype=torch.float32)
                )
                if ln.bias is not None:
                    b_noise = self._make_antithetic_flat(
                        torch.randn(half_size, ln.bias.shape[0], device=device, dtype=torch.float32)
                    )
                else:
                    b_noise = None
                ln.set_noise(w_noise, b_noise)

    def compute_update(self, linear_layers, fitness, avg_loss):
        if not hasattr(self, 'ln_layers'):
            raise RuntimeError("LayerGroupedEGGROLL requires ln_layers to be set")
        ln_layers = self.ln_layers
        groups = self._build_groups(linear_layers, ln_layers)
        N = self.pop_size
        alpha = self.alpha
        sigma = self.sigma

        if self.generation == 0:
            for layer in linear_layers:
                A = layer.A
                B = layer.B
                delta = (alpha / (N * sigma)) * torch.einsum('n,nor,nir->oi', fitness, A, B)
                assert delta.shape == layer.M.shape
                layer.M.data += delta
                layer.set_population(None, None)
            for ln in ln_layers:
                if ln.weight_noise is not None:
                    scale = 1.0 / ln.weight.shape[0]
                    delta = scale * (alpha / (N * sigma)) * (fitness[:, None] * ln.weight_noise).sum(0)
                    ln.weight.data += delta
                if ln.bias_noise is not None:
                    scale = 1.0 / ln.weight.shape[0]
                    delta = scale * (alpha / (N * sigma)) * (fitness[:, None] * ln.bias_noise).sum(0)
                    ln.bias.data += delta
                ln.set_noise(None, None)
            if hasattr(self, 'wpe_module') and self.wpe_module is not None and self.wpe_module.wpe_noise is not None:
                scale = 1.0 / self.wpe_module.config.n_embd
                delta = scale * (alpha / (N * sigma)) * torch.einsum('n,ntc->tc', fitness, self.wpe_module.wpe_noise)
                self.wpe_module.transformer.wpe.weight.data += delta
                self.wpe_module.set_wpe_noise(None)
        else:
            group_idx = (self.generation - 1) % len(groups)
            active_lins, active_lns = groups[group_idx]

            for layer in active_lins:
                A = layer.A
                B = layer.B
                delta = (alpha / (N * sigma)) * torch.einsum('n,nor,nir->oi', fitness, A, B)
                assert delta.shape == layer.M.shape
                layer.M.data += delta
                layer.set_population(None, None)
            for ln in active_lns:
                if ln.weight_noise is not None:
                    scale = 1.0 / ln.weight.shape[0]
                    delta = scale * (alpha / (N * sigma)) * (fitness[:, None] * ln.weight_noise).sum(0)
                    ln.weight.data += delta
                if ln.bias_noise is not None:
                    scale = 1.0 / ln.weight.shape[0]
                    delta = scale * (alpha / (N * sigma)) * (fitness[:, None] * ln.bias_noise).sum(0)
                    ln.bias.data += delta
                ln.set_noise(None, None)
