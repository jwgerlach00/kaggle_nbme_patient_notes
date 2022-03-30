from turtle import forward
import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim*heads == self.embed_size, 'embed_size must be divisible by heads'
        
        # Linear layers
        self.values, self.keys, self.queries = [nn.Linear(self.head_dim, self.head_dim, bias=False) for _ in 
                                                [self.values, self.keys, self.queries]]
        self.fc_out = nn.Linear(heads*self.head_dim, self.embed_size)
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into heads
        values, keys, queries = [x.reshape(N, value_len, self.heads, self.head_dim) for x in [values, keys, query]]
        
        energy = torch.einsum('nqhd,nkhd->nhqk', (queries, keys))
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
