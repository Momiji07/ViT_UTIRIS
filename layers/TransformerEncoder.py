import torch.nn as nn
from layers.MultiHeadAttention import MultiHeadAttention
from layers.MLP import MLP

class TransformerEncoder(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim, depth):
        """ [input]
            - dim (int) : 各パッチのベクトルが変換されたベクトルの長さ（参考[1] (1)式 D）
            - depth (int) : Transformer Encoder の層の深さ（参考[1] (2)式 L）
            - n_heads (int) : Multi-Head Attention の head の数
            - mlp_dim (int) : MLP の隠れ層のノード数
        """
        super().__init__()

        # Layers
        self.norm = nn.LayerNorm(dim)
        self.multi_head_attention = MultiHeadAttention(dim = dim, n_heads = n_heads)
        self.mlp = MLP(dim = dim, hidden_dim = mlp_dim)
        self.depth = depth

    def forward(self, x):
        """[input]
            - x (torch.Tensor)
                - x.shape = torch.Size([batch_size, n_patches + 1, dim])
        """
        for _ in range(self.depth):
            x = self.multi_head_attention(self.norm(x)) + x
            x = self.mlp(self.norm(x)) + x

        return x