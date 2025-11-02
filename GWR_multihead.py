from __future__ import division

import argparse

import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import networkx as nx

from mnist_dataset_class import MNIST
from gwr import gwr3


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


class Task_Multi_Net(nn.Module):
    def __init__(self):
        super(Task_Multi_Net, self).__init__()
        self.multi_fcs = nn.ModuleDict({})

    def forward(self, x, choice):
        x = self.multi_fcs[str(choice)](x)
        return x


def train_MultiNet(model, criterion, optimizer, scheduler, feature_inputs, labels, task_id, num_epochs, num_classes_in_task):
    model.train()
    labels = labels.long() % num_classes_in_task
    for epoch in range(1, num_epochs + 1):
        print('classifier epoch----- : ', epoch)
        feature_inputs = feature_inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(feature_inputs, task_id)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


def test_MultiNet(model, criterion, optimizer, scheduler, feature_inputs, labels, pred_tasks, num_of_classes, num_classes_in_task):
    model.eval()
    num_correct = 0
    number_of_correct_by_class = np.zeros(num_of_classes, dtype=int)
    number_by_class = np.zeros(num_of_classes, dtype=int)
    pred_tasks = pred_tasks.to(device)
    for i, feature in enumerate(feature_inputs):
        feature = feature.to(device)
        label = labels[i]
        pred_task = pred_tasks[i]
        with torch.no_grad():
            outputs = model(feature, int(pred_task))
            _, pred = torch.max(outputs, 0)
            pred = num_classes_in_task * pred_task + pred
            if int(pred) == int(label):
                num_correct += 1
                number_of_correct_by_class[int(label)] += 1
            number_by_class[int(label)] += 1
    return num_correct / len(labels), number_of_correct_by_class / number_by_class


act_thr = np.exp(-30)
fir_thr = 0.05
eps_b = 0.05
eps_n = 0.01
tau_b = 3.33
tau_n = 14.3
alpha_b = 1.05
alpha_n = 1.05
h_0 = 1
sti_s = 1
lab_thr = 0.5
max_age = 1000
max_size = 5000
random_state = None
gwr_epochs = 10
gwr_imgs_per_task = 2500

num_classes_in_task = 5
number_of_tasks = 2
num_train_per_class = 5000
num_test_per_class = 1000
batch_size = 10
test_batch_size = 10
classifierNet_epoches = 10
learning_rate = 0.001
step_size = 7
gamma = 1


def parse_args():
    parser = argparse.ArgumentParser(description='GWR multihead incremental MNIST experiment')
    parser.add_argument('--class-by-class-dir', required=True, help='Path to per-class MNIST directory')
    parser.add_argument('--train-file', required=True, help='Path to cumulative MNIST train tensor file')
    parser.add_argument('--test-file', required=True, help='Path to cumulative MNIST test tensor file')
    return parser.parse_args()


def main():
    args = parse_args()
    mnist_dataset = MNIST(args.class_by_class_dir, args.train_file, args.test_file)

    MultiNet = Task_Multi_Net()
    criterion = nn.CrossEntropyLoss()

    model = MNIST_Net()
    model.load_state_dict(torch.load('mnist_sample_model.pt'), strict=False)

    g1 = gwr3(
        act_thr,
        fir_thr,
        eps_b,
        eps_n,
        tau_b,
        tau_n,
        alpha_b,
        alpha_n,
        h_0,
        sti_s,
        lab_thr,
        max_age,
        max_size,
        random_state=None,
    )

    test_data = []

    for task_id in range(number_of_tasks):
        print('###   Train on task ', task_id + 1)

        MultiNet.multi_fcs.update([[str(task_id), nn.Linear(2 * 2 * 50, num_classes_in_task)]])
        MultiNet = MultiNet.to(device)
        optimizer = optim.SGD(MultiNet.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        train_data = []
        for i in range(num_classes_in_task * task_id, num_classes_in_task * (task_id + 1)):
            train_data += mnist_dataset.get_train_by_task(mode, task_id=i)[0:num_train_per_class]
            test_data += mnist_dataset.get_test_by_task(mode, task_id=i)[0:num_test_per_class]

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

        output = torch.tensor([]).to(device)
        labels = torch.tensor([])

        model.eval()
        model = model.to(device)

        for inputs, label in train_loader:
            images = []
            for image in inputs:
                img = transform(image.numpy())
                img = img.transpose(0, 2).transpose(0, 1)
                images.append(img)
            inputs = torch.stack(images).to(device)
            with torch.no_grad():
                output = torch.cat((output, model(inputs)))
                labels = torch.cat((labels, label.float()))

        output1 = output.cpu().numpy()
        output1 = output1[:gwr_imgs_per_task]
        labels1 = labels.numpy()
        labels1 = labels1[:gwr_imgs_per_task]
        labels1 = np.floor(labels1 / num_classes_in_task)

        if task_id == 0:
            graph_gwr1 = g1.train(output1, labels1, n_epochs=gwr_epochs)
        else:
            graph_gwr1 = g1.train(output1, labels1, n_epochs=gwr_epochs, warm_start=True)

        num_nodes1 = graph_gwr1.number_of_nodes()
        print('number of nodes: ', num_nodes1)

        train_MultiNet(
            MultiNet,
            criterion,
            optimizer,
            scheduler,
            output,
            labels,
            task_id,
            classifierNet_epoches,
            num_classes_in_task,
        )

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

    output_test = torch.tensor([]).to(device)
    labels_test = torch.tensor([])

    for inputs, label in test_loader:
        images = []
        for image in inputs:
            img = transform(image.numpy())
            img = img.transpose(0, 2).transpose(0, 1)
            images.append(img)
        inputs = torch.stack(images).to(device)
        with torch.no_grad():
            output_test = torch.cat((output_test, model(inputs)))
            labels_test = torch.cat((labels_test, label.float()))

    output_test1 = output_test.cpu().numpy()
    labels_test1 = labels_test.numpy()

    labels_test1 = np.floor(labels_test1 / num_classes_in_task)

    nodes_per_tasks = g1.nodes_per_task(number_of_tasks)
    print('nodes_per_task : ', nodes_per_tasks)

    acc_g, class_by_class_acc_g = g1.test(output_test1, labels_test1, number_of_tasks)
    print('GWR classification overall accuracy : ', acc_g)
    print('GWR class_by_class accuracy : ', class_by_class_acc_g)

    pred_tasks = g1.choose_task(output_test1, number_of_tasks)
    pred_tasks = torch.from_numpy(pred_tasks)

    acc, class_by_class_acc = test_MultiNet(
        MultiNet,
        criterion,
        optimizer,
        scheduler,
        output_test,
        labels_test,
        pred_tasks,
        number_of_tasks * num_classes_in_task,
        num_classes_in_task,
    )
    print('GWR Multihead classification overall accuracy : ', acc)
    print('GWR Multihead class_by_class accuracy : ', class_by_class_acc)


if __name__ == '__main__':
    main()
