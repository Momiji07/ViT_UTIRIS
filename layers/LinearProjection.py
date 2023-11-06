import torch.nn as nn

class LinearProjection(nn.Module):
    def __init__(self, patch_dim, dim):
        """ [input]
            - patch_dim (int) : 一枚あたりのパッチのベクトルの長さ（= channels * (patch_size ** 2)）
            - dim (int) : パッチのベクトルが変換されたベクトルの長さ 
        """
        super().__init__()
        self.net = nn.Linear(patch_dim, dim)

    def forward(self, x):
        """ [input]
            - x (torch.Tensor) 
                - x.shape = torch.Size([batch_size, n_patches, patch_dim])
        """
        x = self.net(x)
        return x