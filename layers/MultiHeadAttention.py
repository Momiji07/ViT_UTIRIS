import torch
import torch.nn as nn

from einops import repeat
from einops.layers.torch import Rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        """ [input]
            - dim (int) : パッチのベクトルが変換されたベクトルの長さ
            - n_heads (int) : heads の数
        """
        super().__init__()
        self.n_heads = n_heads
        self.dim_heads = dim // n_heads

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        self.split_into_heads = Rearrange("b n (h d) -> b h n d", h = self.n_heads)

        self.softmax = nn.Softmax(dim = -1)

        self.concat = Rearrange("b h n d -> b n (h d)", h = self.n_heads)

    def forward(self, x):
        """[input]
            - x (torch.Tensor)
                - x.shape = torch.Size([batch_size, n_patches + 1, dim])
        """
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = self.split_into_heads(q)
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        # Logit[i] = Q[i] * tK[i] / sqrt(D) (i = 1, ... , n_heads)
        # AttentionWeight[i] = Softmax(Logit[i]) (i = 1, ... , n_heads)
        logit = torch.matmul(q, k.transpose(-1, -2)) * (self.dim_heads ** -0.5)
        attention_weight = self.softmax(logit)

        # Head[i] = AttentionWeight[i] * V[i] (i = 1, ... , n_heads)
        # Output = concat[Head[1], ... , Head[n_heads]]
        output = torch.matmul(attention_weight, v)
        output = self.concat(output)
        return output
