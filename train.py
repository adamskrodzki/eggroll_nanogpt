"""
EGGROLL training script for nanoGPT.
No backpropagation, no Adam – uses evolution strategies with low‑rank perturbations.
Assumes single GPU (no DDP) for simplicity.
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, GPT, EGGROLLinear

# -----------------------------------------------------------------------------
# EGGROLL hyperparameters
pop_size = 1024          # number of population members per iteration (N)
rank = 1                 # rank of perturbations (r)
sigma = 0.01             # ES noise standard deviation
alpha = 0.1              # learning rate (step size)
max_iters = 100000
eval_interval = 2000
log_interval = 10
eval_iters = 200
# -----------------------------------------------------------------------------
# Data / model config (same as original nanoGPT)
out_dir = 'out'
dataset = 'openwebtext'
batch_size = 12
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False   # compilation not tested with dynamic shapes in EGGROLL
# -----------------------------------------------------------------------------

# Setup
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loader (unchanged)
data_dir = os.path.join('data', dataset)
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Model initialisation
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=None, dropout=dropout, sigma=sigma, rank=rank
)
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    model_args['vocab_size'] = meta_vocab_size
else:
    model_args['vocab_size'] = 50304

config = GPTConfig(**model_args)
model = GPT(config).to(device)

# Collect all EGGROLLinear layers (including lm_head)
linear_layers = [module for module in model.modules() if isinstance(module, EGGROLLinear)]

# Helper to evaluate loss on validation set (using mean parameters, no perturbation)
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    # Make sure no population factors are set
    for layer in linear_layers:
        layer.set_population(None, None)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)   # loss is scalar
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training loop
iter_num = 0
best_val_loss = 1e9
os.makedirs(out_dir, exist_ok=True)
t0 = time.time()

while iter_num < max_iters:
    # Get a fresh batch of data (single batch, no population)
    X, Y = get_batch('train')   # (B, T)

    # --- Step 1: Sample population factors for all linear layers ---
    N = pop_size
    for layer in linear_layers:
        # For rank = 1
        A = torch.randn(N, layer.out_features, 1, device=device, dtype=torch.float32)
        B = torch.randn(N, layer.in_features, 1, device=device, dtype=torch.float32)
        layer.set_population(A, B)

    # --- Step 2: Evaluate all N members ---
    # Expand data to (N, B, T)
    X_pop = X.unsqueeze(0).expand(N, -1, -1)     # (N, B, T)
    Y_pop = Y.unsqueeze(0).expand(N, -1, -1)     # (N, B, T)

    with ctx:
        logits, loss_per_member = model(X_pop, Y_pop)   # loss_per_member shape (N,)
    # Fitness: we want to maximise, so fitness = -loss
    fitness = -loss_per_member.detach()          # (N,)

    # --- Step 3: Fitness shaping (centred) ---
    fitness = fitness - fitness.mean()           # zero mean

    # --- Step 4: Aggregate updates for each layer ---
    for layer in linear_layers:
        A = layer.A.squeeze(-1)   # (N, out_features)
        B = layer.B.squeeze(-1)   # (N, in_features)
        # delta = (alpha / N) * A^T diag(fitness) B
        delta = (alpha / N) * (A.T @ (fitness.unsqueeze(1) * B))   # (out_features, in_features)
        layer.M.data += delta
        # Bias is not updated in this simple version (bias would need separate treatment)
        # Clear factors to free memory
        layer.set_population(None, None)

    # --- Logging and evaluation ---
    if iter_num % log_interval == 0:
        dt = time.time() - t0
        t0 = time.time()
        mean_loss = loss_per_member.mean().item()
        print(f"iter {iter_num}: train loss {mean_loss:.4f}, fitness mean {fitness.mean().item():.4f}, time {dt*1000:.2f}ms")

    if iter_num % eval_interval == 0:
        val_loss = estimate_loss()
        print(f"step {iter_num}: train loss {mean_loss:.4f}, val loss {val_loss['val']:.4f}")
        if val_loss['val'] < best_val_loss:
            best_val_loss = val_loss['val']
            checkpoint = {
                'model': model.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            print(f"saved checkpoint to {out_dir}/ckpt.pt")

    iter_num += 1

print("Training finished.")