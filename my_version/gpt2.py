import os
import time
import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from hellaswag import render_example, iterate_examples
#----------------------------------------------------

class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # because of multi-head attention
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd) # 3 for key, query, value
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # final projection
        self.c_proj.SCALE = 1.0 # for scaling the output
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
    def forward(self, x):
        B, T, C = x.size() # batch, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dimension
        # nh is the number of heads, hs is the head size, and C (number of channels) = nh * hs, is the embedding dimensionality
        # e.g. in GPT-2 (124M), nh=12, hs=64, nh*hs = C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2) # split function is used to split the tensor into 3 parts
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # transpose to have the batch dimension first (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scale_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
    
class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd*4) # fc stands for fully connected
        self.gelu = nn.GELU() # activation function
        self.c_proj = nn.Linear(config.n_embd*4, config.n_embd)
        self.c_proj.SCALE = 1.0 # for scaling the output
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
@dataclass
class GPTConfig:
    block_size: int = 1024 # maximum sequence length
    n_layer: int = 12 # number of transformer blocks
    n_head: int = 12  # number of attention heads
    n_embd: int = 768 # embedding dimensionality
    vocab_size: int = 50257 # number of tokens in the vocabulary: 50000 BPE merges + 256 bytes tokens + 1 <endoftext> token
    
class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), 
            wpe = nn.Embedding(config.block_size, config.n_embd), 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]) 
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # output embedding

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init weights
        self.apply(self._init_weights)