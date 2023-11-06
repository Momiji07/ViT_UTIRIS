from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc
from models.ViT import ViT
from data.ImageLoad import ImageLoad

# データセットの読み込み
train_dataloader, test_dataloader = ImageLoad("data/UTIRIS_V.1/RGB_Images/").load()

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
    n_classes=10,
    dim=64,
    depth=6,
    n_heads=4,
    channels=3,
    mlp_dim=128
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

# 損失関数
criterion = nn.CrossEntropyLoss()

# 最適化手法
optimizer = optim.Adam(net.parameters(), lr=0.001)

# トレーニングループ
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 勾配の初期化
        optimizer.zero_grad()

        # 順伝播
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # 逆伝播
        loss.backward()
        optimizer.step()

# テストデータでの推定
with torch.no_grad():
    outputs = net(test_dataloader)
    _, predicted = torch.max(outputs.data, 1)

# FRRとEERの計算
y_true = np.array(test_dataloader.dataset.targets)
y_pred = np.array(predicted.cpu())

# FRRの計算
frr = 1 - accuracy_score(y_true, y_pred)

# EERの計算
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
eer = 1 - np.max(0.5 * (np.abs(tpr - fpr) + np.abs(tpr - (1 - fpr))))

# 混同行列の計算
conf_matrix = confusion_matrix(y_true, y_pred)

print(f"FRR: {frr}")
print(f"EER: {eer}")
print(f"Confusion Matrix: \n{conf_matrix}")