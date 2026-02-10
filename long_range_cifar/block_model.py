######################
#bnkausik at gmail.com
#code for the recurrent tranformers in the paper https://arxiv.org/abs/2402.14746
######################
#recursive transformer for selective copying

import math
import numpy as np
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from config import training_config, GPTConfig



class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

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
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        #self.attn_dropout = nn.Dropout(config.dropout)  # not req'd with sdpa
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        d_p = self.dropout if self.training else 0

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
        # dropout set to 1e-8 always to force spda to accept dim_v different from dim_q
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=d_p, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)
        self.relu    = nn.ReLU() # relu rather than gelu to combat vanishing gradients
        self.c_proj  = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
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

        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.lm_head = nn.Linear(config.n_embd, 10, bias=False) # output has 10 possibilities


        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters:", self.get_num_params())

    def get_num_params(self):
        #Return the number of parameters in the model.
        n_params = sum(p.numel() for p in self.parameters())

        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, w = 1):
        device = idx.device
        b, t = idx.size()

        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = (self.transformer.wte(idx)) # token embeddings
        pos = torch.arange(0,t, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos) # position embeddings
        x = self.transformer.drop(tok_emb + pos_emb)

        if w==0:  #for regular transformer
            for block in self.transformer.h:
                x = block(x)

        else: # recurrent transformer
            alpha = torch.clamp(self.alpha, 0, 1)
            for block in self.transformer.h:
                x1 = torch.cat((x[:,0:w],block(x[:,0:w])),dim=1)
                i = w
                while True:
                    if i+w <=  t:
                        #handle full block
                        x1 = torch.cat((alpha*x1[:,0:w]+x1[:,w:],alpha*x1[:,0:w]+x[:,i:i+w]),dim=1)
                    else:
                        #handle remnant block
                        x1 = torch.cat((alpha*x1[:,0:w]+x1[:,w:],alpha*x1[:,0:t-i]+x[:,i:]),dim=1)
                    x1 = block(x1)
                    if i == w:
                        x2 = torch.clone(x1[:,0:w])
                    elif i+w < t:
                        x2 = torch.cat((x2,x1[:,0:w]),dim=1)
                    else:
                        x2 = torch.cat((x2,x1[:,0:w]),dim=1)
                        x2 = torch.cat((x2,block(x1[:,w:])),dim=1)
                        break
                    i += w
                x = x2


        x = self.transformer.ln_f(x)
        x = x[:,-1,:]
        logits = self.lm_head(x)
        outputs = torch.argmax(logits,dim=1)[:,None]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return outputs, loss


