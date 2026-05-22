import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):


    def pos_encoding(self, seq_len, d_model):
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        pe = torch.zeros(seq_len, d_model, dtype=torch.float32)
        div_term = torch.pow(
        10000.0,
        torch.arange(0, d_model, 2, dtype=torch.float32) / d_model
        )
        pe[:, 0::2] = torch.sin(pos / div_term)
        pe[:, 1::2] = torch.cos(pos / div_term)
        return torch.round(pe, 5)

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.attn_dim = torch.tensor(attention_dim, dtype=torch.float)
        self.w_k = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.w_q = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.w_v = nn.Linear(embedding_dim, attention_dim, bias=False)
        pass

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        k = self.w_k(embedded)
        q = self.w_q(embedded)
        v = self.w_v(embedded)
        attn = torch.einsum("bqd,bkd->bqk", q, k) / torch.sqrt(self.attn_dim)
        mask = torch.tril(torch.ones(attn.size(-2), attn.size(-1)))
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = nn.functional.softmax(attn, dim=2)
        hidden = torch.einsum("bqk,bkv->bqv", attn, v)
        return hidden
        
