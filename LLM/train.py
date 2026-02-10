######################
#bnkausik at gmail.com
#code for the recurrent tranformers in the paper https://arxiv.org/abs/2402.14746
######################

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O

out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# data
dataset = 'openwebtext'
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

# learning rate decay settings
#lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
decay_lr = True


# system
device= 'mps'
dtype = 'float32'
device_type = 'cpu'

###################################my tweaks

#override parameters
rec_block = 32
eval_only = False
warmup_iters = 0
vocab_size = 50304
target_loss = 3.3





# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
# if not ddp, we are running on a single gpu, and one process
master_process = True
tokens_per_iter = batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

#torch.manual_seed(1337 + seed_offset)   # my tweak not fixing seed!!!!

# poor man's data loader
data_dir = os.path.join('../data', dataset) # tweaked to centralize data

@torch.no_grad()
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122

    if split == 'val':
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])


    x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  rec_block = rec_block,# my tweak
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    #checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only = False)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'bias', 'vocab_size']:  #my tweak  suppress block size
    #for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']

model.to(device)


# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    out_stderr={}
    model.eval();
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        out_stderr[split]=losses.std()/math.sqrt(eval_iters)
    model.train()
    return out,out_stderr


print(config) # output all config parameters
print(" overrides",model_args)
sys.stdout.flush()

# estimate starting losses
if iter_num==0:
    cur_lr = optimizer.param_groups[0]['lr']
    losses,std_err = estimate_loss()
    if rec_block==0:
        print(f" iter_e {iter_num} tokens {iter_num*batch_size*block_size:3e} lr {cur_lr:.3e} train_loss {losses['train']:.3f} val_loss {losses['val']:.3f} {std_err['train']:.3e} {std_err['val']:.3e}")
    else:
        print(f" iter_e {iter_num} tokens {iter_num*batch_size*block_size:3e} lr {cur_lr:.3e} train_loss {losses['train']:.3f} val_loss {losses['val']:.3f} {std_err['train']:.3e} {std_err['val']:.3e} alpha {model.alpha.item():.2f} ")
    sys.stdout.flush()
    if eval_only: sys.exit()

# training loop
#get first batch
X, Y = get_batch('train')
model.train()



while True:

    if iter_num % eval_interval == 0 and iter_num >0:

        #save good state
        if always_save_checkpoint:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'config': config,
            }
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            #print("saved checkpoint")

        cur_lr = optimizer.param_groups[0]['lr']
        losses,std_err = estimate_loss()
        if rec_block==0:
            print(f" iter_e {iter_num} tokens {iter_num*batch_size*block_size:3e} lr {cur_lr:.3e} train_loss {losses['train']:.3f} val_loss {losses['val']:.3f} {std_err['train']:.3e} {std_err['val']:.3e}")
        else:
            print(f" iter_e {iter_num} tokens {iter_num*batch_size*block_size:3e} lr {cur_lr:.3e} train_loss {losses['train']:.3f} val_loss {losses['val']:.3f} {std_err['train']:.3e} {std_err['val']:.3e} alpha {model.alpha.item():.2f} ")
        sys.stdout.flush()

        # set learning rate
        if decay_lr:
            new_lr = np.clip(learning_rate*(1-np.exp(target_loss-losses['val'])),min_lr,learning_rate)
            for param_group in optimizer.param_groups: param_group['lr'] = new_lr


    #now do some training
    loss = model(X, Y)

    # immediately async prefetch next batch while model is doing the forward pass on the GPU
    X, Y = get_batch('train')

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()


    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
