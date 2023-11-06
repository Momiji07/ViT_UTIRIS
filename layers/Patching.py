import torch.nn as nn

from einops.layers.torch import Rearrange

class Patching(nn.Module):
  def __init__(self, patch_height, patch_width):
    """ [input]
        - patch_size (int) : パッチの縦の長さ（=横の長さ）
    """
    super().__init__()
    self.net = Rearrange("b c (h ph) (w pw) -> b (h w) (ph pw c)", ph = patch_height, pw = patch_width)

  def forward(self, x):
    """ [input]
    - x (torch.Tensor) : 画像データ
        - x.shape = torch.Size([batch_size, channels, image_height, image_width])
    """
    print(f'x shape: {x.shape}')  # 追加したデバッグステートメント
    return self.net(x)