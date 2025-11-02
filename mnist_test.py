from __future__ import division

import argparse

import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch.utils.data import DataLoader

from mnist_dataset_class import MNIST
from gwr import gwr

mode = 'incremental_nc'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5, 1)
        self.conv2 = nn.Conv2d(10, 20, 5, 1)
        self.conv3 = nn.Conv2d(20, 50, 3, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 2 * 2 * 50)
        return x


def parse_args():
    parser = argparse.ArgumentParser(description='GWR clustering test on MNIST features')
    parser.add_argument('--class-by-class-dir', required=True, help='Path to per-class MNIST directory')
    parser.add_argument('--train-file', required=True, help='Path to cumulative MNIST train tensor file')
    parser.add_argument('--test-file', required=True, help='Path to cumulative MNIST test tensor file')
    return parser.parse_args()


def extract_features(model, loader):
    features = torch.tensor([]).to(device)
    labels = torch.tensor([])
    model = model.to(device)
    model.eval()
    for inputs, label in loader:
        images = []
        for image in inputs:
            img = transform(image.numpy())
            img = img.transpose(0, 2).transpose(0, 1)
            images.append(img)
        batch = torch.stack(images).to(device)
        with torch.no_grad():
            features = torch.cat((features, model(batch)))
            labels = torch.cat((labels, label.float()))
    return features.cpu().numpy(), labels.numpy()


def main():
    args = parse_args()
    mnist_dataset = MNIST(args.class_by_class_dir, args.train_file, args.test_file)

    train_samples = []
    for task_id in range(10):
        train_samples += mnist_dataset.get_train_by_task(mode, task_id=task_id)[:500]
    train_loader = DataLoader(train_samples, batch_size=100, shuffle=True)
    print('number of train samples : ', len(train_samples))

    model = MNIST_Net()
    model.load_state_dict(torch.load('mnist_sample_model.pt'), strict=False)

    train_features, train_labels = extract_features(model, train_loader)

    epochs = 5
    g1 = gwr(act_thr=0.70, fir_thr=0.1, random_state=None, max_size=5000)
    graph_gwr = g1.train(train_features, train_labels, n_epochs=epochs)
    number_of_clusters = nx.number_connected_components(graph_gwr)
    num_nodes = graph_gwr.number_of_nodes()
    print('approach 1 ......')
    print('number of nodes: ', num_nodes)
    print('number of clusters: ', number_of_clusters)

    test_samples = []
    for task_id in range(10):
        test_samples += mnist_dataset.get_test_by_task(mode, task_id=task_id)
    test_loader = DataLoader(test_samples, batch_size=100, shuffle=True)

    test_features, test_labels = extract_features(model, test_loader)

    acc, class_by_class_acc = g1.test(test_features, test_labels)
    print('gwr clustering overall accuracy : ', acc)
    print('gwr clustering class_by_class accuracy : ', class_by_class_acc)

    acc_knn, class_by_class_acc_knn = g1.KNearest_test(test_features, test_labels)
    print('Using K-Nearest ..')
    print('gwr clustering overall accuracy : ', acc_knn)
    print('gwr clustering class_by_class accuracy : ', class_by_class_acc_knn)


if __name__ == '__main__':
    main()
