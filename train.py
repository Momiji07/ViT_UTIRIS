from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc
from models.ViT import ViT
from data.ImageLoad import ImageLoad

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \\n")

# データセットの読み込み
train_dataloader, test_dataloader = ImageLoad("data/RGB_Images/").load()

""" [input]
    - image_size (int) : 画像の縦の長さ（= 横の長さ）
    - patch_size (int) : パッチの縦の長さ（= 横の長さ）
    - n_classes (int) : 分類するクラスの数
    - dim (int) : 各パッチのベクトルが変換されたベクトルの長さ（参考[1] (1)式 D）
    - depth (int) : Transformer Encoder の層の深さ（参考[1] (2)式 L）
    - n_heads (int) : Multi-Head Attention の head の数
    - chahnnels (int) : 入力のチャネル数（RGBの画像なら3）
    - mlp_dim (int) : MLP の隠れ層のノード数
"""

# モデルの定義
net = ViT(
    image_height=170,
    image_width=256,
    patch_height=5,
    patch_width=16,
    n_classes=79,
    dim=64,
    depth=6,
    n_heads=4,
    channels=3,
    mlp_dim=128
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)

# トレーニングループ
num_epochs = 1
for epoch in range(num_epochs):
  print(f"Epoch {epoch+1}\\n-------------------------------")
  train(train_dataloader, net, loss_fn, optimizer)
  test(test_dataloader, net, loss_fn)
print("Done!")