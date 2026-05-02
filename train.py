"""
Training script for character-level language modeling using EGGROLL
(Evolutionary Gradient generation with GRadient-fRee Optimization for Large-scale Learning).

To train on Shakespeare characters:
$ python train.py config/train_shakespeare_char.py
"""

import os
import time
import pickle

import numpy as np
import torch

from model import GPTConfig, GPT, EGGROLLinear, LayerNorm
from strategies import STRATEGY_REGISTRY

# -----------------------------------------------------------------------------
# default config values (designed for character-level Shakespeare)
out_dir = 'out'
eval_interval = 250
log_interval = 10
eval_iters = 20
always_save_checkpoint = False
dataset = 'shakespeare_char'
accumulation_steps = 1
batch_size = 16
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False
pop_size = 128
rank = 1
sigma = 0.01
alpha = 0.1
max_iters = 5000
strategy = 'standard'
device = 'cuda'
compile = True
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

device_type = 'cuda' if 'cuda' in device else 'cpu'
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

os.makedirs(out_dir, exist_ok=True)

# poor man's data loader
data_dir = os.path.join('data', dataset)

def get_batch(split):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']

# model init
iter_num = 0
best_val_loss = 1e9

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, sigma=sigma, rank=rank)
model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 65
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

# EGGROLL: collect all linear layers
linear_layers = [module for module in model.modules() if isinstance(module, EGGROLLinear)]
ln_layers = [module for module in model.modules() if isinstance(module, LayerNorm)]

# EGGROLL: instantiate update strategy
egroll_strategy = STRATEGY_REGISTRY[strategy](alpha=alpha, sigma=sigma, rank=rank, pop_size=pop_size)
egroll_strategy.ln_layers = ln_layers
egroll_strategy.wpe_module = model

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# helps estimate an accurately averaged loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for layer in linear_layers:
        layer.set_population(None, None)
    for ln in ln_layers:
        ln.set_noise(None, None)
    model.set_wpe_noise(None)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# -----------------------------------------------------------------------------
# EGGROLL training loop
# -----------------------------------------------------------------------------
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0

while True:
    egroll_strategy.sample_population(linear_layers, device)

    acc_loss = torch.zeros(pop_size, device=device)

    for step in range(accumulation_steps):
        X, Y = get_batch('train')
        X_pop = X.unsqueeze(0).expand(pop_size, -1, -1)
        Y_pop = Y.unsqueeze(0).expand(pop_size, -1, -1)
        with torch.no_grad():
            _, loss_per_member = model(X_pop, Y_pop)
        acc_loss += loss_per_member

    avg_loss = acc_loss / accumulation_steps
    fitness = egroll_strategy._compute_fitness(avg_loss)

    egroll_strategy.compute_update(linear_layers, fitness, avg_loss)
    egroll_strategy.on_generation_end(avg_loss)

    # logging
    if iter_num % log_interval == 0:
        lossf = loss_per_member.mean().item()
        if device_type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        best_idx = torch.argmax(acc_loss).item()
        if iter_num >= 5:
            steps_log = 1 if iter_num == 0 else log_interval
            dt_avg = dt / steps_log
            mfu = model.estimate_mfu(batch_size * accumulation_steps, dt_avg, pop_size)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt_avg*1000:.2f}ms, mfu {running_mfu*100:.2f}%, min loss {acc_loss[best_idx].item():.4f}")
        else:
            print(f"iter {iter_num}: loss {lossf:.4f}")

    # evaluation and checkpointing
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    # fetch next batch
    X, Y = get_batch('train')

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break
