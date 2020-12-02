from __future__ import division

import sys
import os

import numpy as np
import sklearn.datasets

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch 
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import copy
import networkx as nx

from mnist_dataset_class import *
from gwr import gwr,gwr2


class_by_class_dir = 'D:/Semester7/FYP/code base/datasets/MNIST/class_by_class'
train_dir = 'D:/Semester7/FYP/code base/datasets/MNIST/cumulative/train.pt'
test_dir = 'D:/Semester7/FYP/code base/datasets/MNIST/cumulative/test.pt'
mode = 'incremental_nc'

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
MNIST_dataset = MNIST(class_by_class_dir, train_dir, test_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5, 1)
        self.conv2 = nn.Conv2d(10, 20, 5, 1)
        self.conv3 = nn.Conv2d(20, 50, 3, 1)
        # # self.fc1 = nn.Linear(4*4*50, 256)
        # # self.fc2 = nn.Linear(256, 10)
        # self.fc = nn.Linear(2*2*50, 10)
        # #self.mode = mode
        
    def forward(self, x, replay_data=torch.tensor([]), mode = "Train"):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 2*2*50)
        # x = F.relu(self.fc1(x))
        # x = self.fc(x)
        return x

train_data_ = []
for i in range(10):
	train_data_ += MNIST_dataset.get_train_by_task(mode, task_id = i)[:500]
train_loader = torch.utils.data.DataLoader(train_data_, batch_size = 100, shuffle = True)
print('number of train samples : ',len(train_data_))
# train_data = np.array(train_data_)

model = MNIST_Net()
model.load_state_dict(torch.load('mnist_sample_model.pt'), strict = False)

output = torch.tensor([])
labels = torch.tensor([])
output = output.to(device)
model.eval()
model = model.to(device)

for inputs,label in train_loader:
	images =[]
	for image in inputs:
		img = transform(image.numpy())
		img = img.transpose(0,2).transpose(0,1)
		images.append(img)
	inputs = torch.stack(images)
	inputs = inputs.to(device)
	with torch.no_grad():
		# output.append(model(inputs))
		output = torch.cat((output,model(inputs)))
		labels = torch.cat((labels,label.float()))
output = output.cpu()
output = output.numpy()
labels = labels.numpy()
# print(labels)
# input('.....')
# print(output.shape)
# print(output[1].shape)

epochs = 5

g1 = gwr(act_thr = 0.70, fir_thr = 0.1, random_state=None, max_size=5000)
graph_gwr1 = g1.train(output, labels, n_epochs=epochs)
# graph_gwr = g.train(output, n_epochs=epochs)
# Xg = g.get_positions()
number_of_clusters1 = nx.number_connected_components(graph_gwr1)
# print('number of clusters: ',number_of_clusters1)
num_nodes1 = graph_gwr1.number_of_nodes()
print('number of nodes: ',num_nodes1)

# g2 = gwr2(act_thr = 0.95, fir_thr = 0.1, random_state=None, max_size=5000)
# graph_gwr2 = g2.train(output, labels, n_epochs=epochs)
# # graph_gwr = g.train(output, n_epochs=epochs)
# # Xg = g.get_positions()
# number_of_clusters2 = nx.number_connected_components(graph_gwr2)
# # print('number of clusters: ',number_of_clusters2)
# num_nodes2 = graph_gwr2.number_of_nodes()
# # print('number of nodes: ',num_nodes2)

#testing
test_data_ = []
for i in range(10):
	test_data_ += MNIST_dataset.get_test_by_task(mode, task_id = i)
test_loader = torch.utils.data.DataLoader(test_data_, batch_size = 100, shuffle = True)
# print('number of test samples : ',len(test_data_))

output_test = torch.tensor([])
labels_test = torch.tensor([])
output_test = output_test.to(device)
for inputs,label in test_loader:
	images =[]
	for image in inputs:
		img = transform(image.numpy())
		img = img.transpose(0,2).transpose(0,1)
		images.append(img)
	inputs = torch.stack(images)
	inputs = inputs.to(device)
	with torch.no_grad():
		# output_test.append(model(inputs))
		output_test = torch.cat((output_test,model(inputs)))
		labels_test = torch.cat((labels_test,label.float()))
output_test = output_test.cpu()
output_test = output_test.numpy()
labels_test = labels_test.numpy()

print('approach 1 ......')
print('number of nodes: ',num_nodes1)
print('number of clusters: ',number_of_clusters1)
acc,class_by_class_acc = g1.test(output_test, labels_test)
print('gwr clustering overall accuracy : ', acc)
print('gwr clustering class_by_class accuracy : ', class_by_class_acc)
# np.savez('acc_file1.npz', acc, class_by_class_acc)

print('Using K-Nearest ..')
acc,class_by_class_acc = g1.KNearest_test(output_test, labels_test)
print('gwr clustering overall accuracy : ', acc)
print('gwr clustering class_by_class accuracy : ', class_by_class_acc)
# np.savez('acc_file2.npz', acc, class_by_class_acc)

# print('approach 2 ......')
# print('number of nodes: ',num_nodes2)
# print('number of clusters: ',number_of_clusters2)
# acc,class_by_class_acc = g2.test(output_test, labels_test)
# print('gwr clustering overall accuracy : ', acc)
# print('gwr clustering class_by_class accuracy : ', class_by_class_acc)
# np.savez('acc_file3.npz', acc, class_by_class_acc)

# print('Using K-Nearest ..')
# acc,class_by_class_acc = g2.KNearest_test(output_test, labels_test)
# print('gwr clustering overall accuracy : ', acc)
# print('gwr clustering class_by_class accuracy : ', class_by_class_acc)
# np.savez('acc_file4.npz', acc, class_by_class_acc)

# os.system(r'rundll32.exe powrprof.dll,SetSuspendState Hibernate')

# def generate_feature_samples(model,train_loader):
