import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class EGGROLLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, r=1, sigma=0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.sigma = sigma
        self.M = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.A = None
        self.B = None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.M, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.M)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def set_population(self, A, B):
        self.A = A
        self.B = B

    def forward(self, x):
        if self.A is None or x.dim() == 3:
            out = x @ self.M.T
            if self.bias is not None:
                out += self.bias
            return out
        N, B, T, in_f = x.shape
        base = x[0] @ self.M.T
        if self.bias is not None:
            base += self.bias
        base = base.unsqueeze(0).expand(N, -1, -1, -1)
        x_flat = x.view(N, B*T, in_f)
        tmp = torch.einsum('nbd, nir -> nbr', x_flat, self.B)
        correction = (self.sigma / math.sqrt(self.r)) * torch.einsum('nbr, nor -> nbo', tmp, self.A)
        correction = correction.view(N, B, T, -1)
        return base + correction


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = EGGROLLinear(config.n_embd, 3 * config.n_embd, bias=config.bias, sigma=config.sigma, r=config.rank)
        self.c_proj = EGGROLLinear(config.n_embd, config.n_embd, bias=config.bias, sigma=config.sigma, r=config.rank)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        if x.dim() == 4:
            N, B, T, C = x.shape
            total_batch = N * B
            x = x.view(total_batch, T, C)
            merge_nb = True
        else:
            B, T, C = x.shape
            total_batch = B
            N = 1
            merge_nb = False

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(total_batch, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(total_batch, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(total_batch, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(total_batch, T, C)
        y = self.resid_dropout(self.c_proj(y))

        if merge_nb:
            y = y.view(N, B, T, C)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = EGGROLLinear(config.n_embd, 4 * config.n_embd, bias=config.bias, sigma=config.sigma, r=config.rank)
        self.gelu    = nn.GELU()
        self.c_proj  = EGGROLLinear(4 * config.n_embd, config.n_embd, bias=config.bias, sigma=config.sigma, r=config.rank)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False
    sigma: float = 0.01
    rank: int = 1


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = EGGROLLinear(config.n_embd, config.vocab_size, bias=False, sigma=config.sigma, r=config.rank)
        self.transformer.wte.weight = self.lm_head.M

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        if idx.dim() == 3:
            N, B, T = idx.shape
            has_pop = True
            idx = idx.contiguous()
            if targets is not None:
                targets = targets.contiguous()
        else:
            B, T = idx.shape
            N = 1
            has_pop = False
            idx = idx.unsqueeze(0)
            if targets is not None:
                targets = targets.unsqueeze(0)

        device = idx.device
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(N*B*T, -1), targets.view(N*B*T), ignore_index=-1, reduction='none')
            loss = loss.view(N, B*T).mean(dim=1)
            if not has_pop:
                loss = loss.mean()
            return logits, loss
        else:
            logits = self.lm_head(x[:, :, [-1], :])
            if not has_pop:
                logits = logits.squeeze(0)
            return logits, None

    def estimate_mfu(self, fwdbwd_per_iter, dt, pop_size):
        N = self.get_num_params()
        cfg = self.config
        C, L, T, V = cfg.n_embd, cfg.n_layer, cfg.block_size, cfg.vocab_size
        r = cfg.rank
        sum_f = 16 * C * L + C + V
        flops_per_member = 2 * N + 2 * r * sum_f + 4 * L * C * T
        total_tokens = fwdbwd_per_iter * T * pop_size
        flops_per_iter = total_tokens * flops_per_member
        flops_achieved = flops_per_iter / dt
        flops_promised = 30e12
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
