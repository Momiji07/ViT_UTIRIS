import torch.nn as nn
from layers.Patching import Patching
from layers.LinearProjection import LinearProjection
from layers.Embedding import Embedding
from layers.TransformerEncoder import TransformerEncoder
from layers.MLPHead import MLPHead

class ViT(nn.Module):
    def __init__(self, image_height, image_width, patch_height, patch_width, n_classes, dim, depth, n_heads, channels = 3, mlp_dim = 256):
        """ [input]
            - image_height (int) : 画像の縦の長さ
            - image_width (int) : 画像の横の長さ
            - patch_height (int) : パッチの縦の長さ
            - patch_width (int) : パッチの横の長さ
            - n_classes (int) : 分類するクラスの数
            - dim (int) : 各パッチのベクトルが変換されたベクトルの長さ（参考[1] (1)式 D）
            - depth (int) : Transformer Encoder の層の深さ（参考[1] (2)式 L）
            - n_heads (int) : Multi-Head Attention の head の数
            - chahnnels (int) : 入力のチャネル数（RGBの画像なら3）
            - mlp_dim (int) : MLP の隠れ層のノード数
        """

        super().__init__()
        
        # Params
        n_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.depth = depth

        # Layers
        self.patching = Patching(patch_height = patch_height, patch_width = patch_width)
        self.linear_projection_of_flattened_patches = LinearProjection(patch_dim = patch_dim, dim = dim)
        self.embedding = Embedding(dim = dim, n_patches = n_patches)
        self.transformer_encoder = TransformerEncoder(dim = dim, n_heads = n_heads, mlp_dim = mlp_dim, depth = depth)
        self.mlp_head = MLPHead(dim = dim, out_dim = n_classes)


    def forward(self, img):
        """ [input]
            - img (torch.Tensor) : 画像データ
                - img.shape = torch.Size([batch_size, channels, image_height, image_width])
        """

        x = img

        # 1. パッチに分割
        # x.shape : [batch_size, channels, image_height, image_width] -> [batch_size, n_patches, channels * (patch_size ** 2)]
        x = self.patching(x)

        # 2. 各パッチをベクトルに変換
        # x.shape : [batch_size, n_patches, channels * (patch_size ** 2)] -> [batch_size, n_patches, dim]
        x = self.linear_projection_of_flattened_patches(x)

        # 3. [class] トークン付加 + 位置エンコーディング 
        # x.shape : [batch_size, n_patches, dim] -> [batch_size, n_patches + 1, dim]
        x = self.embedding(x)

        # 4. Transformer Encoder
        # x.shape : No Change
        x = self.transformer_encoder(x)

        # 5. 出力の0番目のベクトルを MLP Head で処理
        # x.shape : [batch_size, n_patches + 1, dim] -> [batch_size, dim] -> [batch_size, n_classes]
        x = x[:, 0]
        x = self.mlp_head(x)

        return x