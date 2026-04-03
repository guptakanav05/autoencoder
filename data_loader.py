import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class EMNISTDataset(Dataset):

    def __init__(self, csv_path):
        print(f"  Loading {csv_path}...")
        try:
            data = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)
        except ValueError:
            data = np.loadtxt(csv_path, delimiter=',', dtype=np.float32, skiprows=1)
        self.labels = torch.tensor(data[:, 0], dtype=torch.long)
        pixels = data[:, 1:]
        pixels = pixels / 255.0
        self.images = torch.tensor(pixels, dtype=torch.float32).reshape(-1, 1, 28, 28)
        self.images = torch.transpose(self.images, 2, 3)
        print(f"  Loaded {len(self.images)} images, shape: {self.images.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_data_loaders(batch_size=128, val_split=0.2, data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(BASE_DIR, 'data')
    print("Loading EMNIST dataset from CSV files...")

    train_dataset = EMNISTDataset(f'{data_dir}/emnist-letters-train.csv')
    test_dataset = EMNISTDataset(f'{data_dir}/emnist-letters-test.csv')

    total_size = len(train_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\nDataset sizes - Train: {train_size}, Validation: {val_size}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_data_loaders()
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"Pixel value range: [{images.min():.2f}, {images.max():.2f}]")
