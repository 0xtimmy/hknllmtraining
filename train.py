import os
import time
import math
import pickle
from contextlib import nullcontext
from tqdm import tqdm

import numpy as np
import torch

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# data
dataset = './dataset'
gradient_accumulation_steps = 1
batch_size = 8
block_size = 256 # context of up to 256 previous characters
logfile = "train.log"

# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

bias = False # do we use bias inside LayerNorm and Linear layers?

# adamw optimizer
learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
beta1 = 0.90 # make a bit bigger because number of tokens per iter is small
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small\

# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
print(config)
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
tokens_per_iter = batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(0)
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

data_dir = dataset
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

# init a new model from scratch
print("Initializing a new model from scratch")
# determine the vocab size we'll use for from-scratch training
model_args['vocab_size'] = 50304
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

model.to(device)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
checkpoint = None # free up memory

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
for iter_num in tqdm(range(max_iters)):
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        model.eval()
        est_losses = {}
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            est_losses[split] = losses.mean()
        model.train()
        print(f"step {iter_num}: train loss {est_losses['train']:.4f}, val loss {est_losses['val']:.4f}")
        if logfile is not None:
            with open(logfile, "a") as f:
                f.write(f"step {iter_num}: train loss {est_losses['train']:.4f}, val loss {est_losses['val']:.4f}")
                f.close()
        if est_losses['val'] < best_val_loss:
            best_val_loss = est_losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                if logfile is not None:
                    with open(logfile, "a") as f:
                        f.write(f"saving checkpoint to {out_dir}")
                        f.close()
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    logits, loss = model(X, Y)
    X, Y = get_batch('train')

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item()
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        if logfile is not None:
            with open(logfile, "a") as f:
                f.write(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms\n")
                f.close()
