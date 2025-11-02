"""
MNIST Data Preparation Script for GWR Multihead
This script downloads MNIST and organizes it into the required format.
"""

import os
import torch
from torchvision import datasets
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, root, train=True, download=True):
        self.mnist = datasets.MNIST(root=root, train=train, download=download)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        return img, label


def prepare_mnist_data(output_dir='.'):
    # Create directories
    class_by_class_train_dir = os.path.join(output_dir, 'class_by_class', 'train')
    class_by_class_test_dir = os.path.join(output_dir, 'class_by_class', 'test')
    os.makedirs(class_by_class_train_dir, exist_ok=True)
    os.makedirs(class_by_class_test_dir, exist_ok=True)

    print('Downloading MNIST data...')
    train_dataset = MNISTDataset(root='./data', train=True, download=True)
    test_dataset = MNISTDataset(root='./data', train=False, download=True)

    # Save cumulative datasets
    train_file = os.path.join(output_dir, 'train.pt')
    test_file = os.path.join(output_dir, 'test.pt')
    torch.save(train_dataset, train_file)
    torch.save(test_dataset, test_file)

    # Create class-by-class splits
    print('Creating class-by-class splits...')
    train_by_class = {i: [] for i in range(10)}
    for idx in range(len(train_dataset)):
        img, label = train_dataset[idx]
        train_by_class[label].append((img, label))

    test_by_class = {i: [] for i in range(10)}
    for idx in range(len(test_dataset)):
        img, label = test_dataset[idx]
        test_by_class[label].append((img, label))

    for class_id in range(10):
        train_class_file = os.path.join(class_by_class_train_dir, f'class_{class_id}.pt')
        torch.save(train_by_class[class_id], train_class_file)

        test_class_file = os.path.join(class_by_class_test_dir, f'class_{class_id}.pt')
        torch.save(test_by_class[class_id], test_class_file)

    print('Data preparation complete.')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare MNIST data for GWR Multihead experiments')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory where data files will be saved')
    args = parser.parse_args()

    prepare_mnist_data(output_dir=args.output_dir)
