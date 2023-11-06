import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class ImgeTransform:
    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.Resize((170, 256)),
            transforms.ToTensor()
        ])

    def __call__(self, img):
        return self.data_transform(img)

class ImageLoad:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def load(self):
        # 元のデータセット
        full_dataset = datasets.ImageFolder(self.dataset_dir, transform=ImgeTransform())

        print(full_dataset.classes)

        # 学習データ、検証データに 8:2 の割合で分割する。
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

        imgs, labels = next(iter(train_dataloader))
        print("Images shape is {}".format(imgs[0].shape))
        print("Labels is {}".format(labels[0]))

        return train_dataloader, test_dataloader

