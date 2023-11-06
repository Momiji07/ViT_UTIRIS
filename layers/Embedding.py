import torch
import torch.nn as nn

from einops import repeat

class Embedding(nn.Module):
    def __init__(self, dim, n_patches):
        """ [input]
            - dim (int) : パッチのベクトルが変換されたベクトルの長さ
            - n_patches (int) : パッチの枚数
        """
        super().__init__()
        # [class] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, dim))
    
    def forward(self, x):
        """[input]
            - x (torch.Tensor)
                - x.shape = torch.Size([batch_size, n_patches, dim])
        """
        # バッチサイズを抽出
        batch_size, _, __ = x.shape

        # [class] トークン付加
        # x.shape : [batch_size, n_patches, patch_dim] -> [batch_size, n_patches + 1, patch_dim]
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b = batch_size)
        x = torch.concat([cls_tokens, x], dim = 1)

        # 位置エンコーディング
        x += self.pos_embedding

        return x