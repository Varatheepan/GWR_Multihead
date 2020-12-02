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
import torch.optim as optim
from torch.optim import lr_scheduler
# from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import copy
import networkx as nx

from mnist_dataset_class import *
from gwr import gwr,gwr2,gwr3


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
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 2*2*50)
        # x = F.relu(self.fc1(x))
        # x = self.fc(x)
        return x

class Task_Multi_Net(nn.Module):
    def __init__(self):
        super(Task_Multi_Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 10, 5, 1)
        # self.conv2 = nn.Conv2d(10, 20, 5, 1)
        # self.conv3 = nn.Conv2d(20, 50, 3, 1)
        # # self.fc1 = nn.Linear(4*4*50, 256)
        # # self.fc2 = nn.Linear(256, 10)
        self.multi_fcs = nn.ModuleDict({}) #'0':nn.Linear(2*2*50, 2)})
        # #self.mode = mode
        
    def forward(self, x, choice):
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv3(x))
        # x = x.view(-1, 2*2*50)
        # x = F.relu(self.fc1(x))
        x = self.multi_fcs[str(choice)](x)
        return x

def train_MultiNet(model, criterion, optimizer, scheduler, feature_inputs, labels, task_id, num_epochs,num_classes_in_task):
	model.train()
	# print(labels)
	labels = labels.long()%num_classes_in_task
	# print(labels)
	for epoch in range(1, num_epochs + 1):
		print('classifier epoch----- : ', epoch)			
		feature_inputs = feature_inputs.to(device)
		labels = labels.to(device)
		optimizer.zero_grad()
		with torch.set_grad_enabled(True):
			outputs = model(feature_inputs,task_id)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
		scheduler.step()
	return model

def test_MultiNet(model, criterion, optimizer, scheduler, feature_inputs, labels, pred_tasks, num_of_classes,num_classes_in_task):
	model.eval()
	num_correct = 0
	#print('lengths: ',len(feature_inputs),len(pred_tasks),len(labels))
	number_of_correct_by_class = np.zeros(num_of_classes, dtype = int)
	number_by_class = np.zeros(num_of_classes, dtype = int)
	pred_tasks = pred_tasks.to(device)
	for i,feature in enumerate(feature_inputs):			
		feature = feature.to(device)
		label = labels[i]
		pred_task = pred_tasks[i]
		with torch.no_grad():
			outputs = model(feature,int(pred_task))
			_, pred = torch.max(outputs,0)
			#print(outputs,pred)
			pred = num_classes_in_task*pred_task + pred
			if(int(pred) == int(label)):
				num_correct += 1
				number_of_correct_by_class[int(label)] += 1
			number_by_class[int(label)] += 1
	return num_correct/len(labels),number_of_correct_by_class/number_by_class

## GWR Parameters ##
#  gwr2 #
# act_thr  = 0.25		
# fir_thr  = 0.05
# eps_b    = 20.0		#moving BM nodes according to activation
# eps_n    = 1.50		#moving neighbour nodes according to activation
# tau_b = 0.5			#Firing
# tau_n = 0.1
# kappa = 1.0
# lab_thr = 0.5
# max_age = 1000
# max_size = 5000
# random_state = None
# gwr_epochs = 10

#  gwr3 #
act_thr  = np.exp(-30)	
fir_thr  = 0.05
eps_b    = 0.05		#moving BM nodes according to activation
eps_n    = 0.01		#moving neighbour nodes according to activation
tau_b = 3.33			#Firing
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

MultiNet = Task_Multi_Net()
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(MultiNet.parameters(), lr = learning_rate, momentum = 0.9)
# scheduler = lr_scheduler.StepLR(optimizer1, step_size = step_size, gamma = gamma)
	
# MultiNet.multi_fcs.update([['0', nn.Linear(2*2*50, 2)]])
# print(MultiNet)
# MultiNet.multi_fcs.update([['1', nn.Linear(2*2*50, 2)]])
# print(MultiNet)
# input('........')

model = MNIST_Net()
model.load_state_dict(torch.load('mnist_sample_model.pt'), strict = False)

#g1 = gwr2(act_thr = 0.5, fir_thr = 0.05, eps_b = 0.5, eps_n = 0.1, random_state=None, max_size= max_nodes)
# g1 = gwr2(act_thr, fir_thr, eps_b,
#                  eps_n, tau_b, tau_n, kappa,
#                  lab_thr, max_age, max_size,
#                  random_state = None)
g1 = gwr3(act_thr, fir_thr, eps_b,
                 eps_n, tau_b, tau_n, alpha_b, alpha_n,
                 h_0, sti_s,lab_thr, max_age, max_size,
                 random_state = None)

test_data = []

for task_id in range(0, number_of_tasks): #, num_classes_in_task):
	print('###   Train on task ', task_id + 1)# str(task_id//num_classes_in_task + 1), '   ###')
	
	MultiNet.multi_fcs.update([[str(task_id), nn.Linear(2*2*50, num_classes_in_task)]])
	print(MultiNet)
	MultiNet = MultiNet.to(device)
	optimizer = optim.SGD(MultiNet.parameters(), lr = learning_rate, momentum = 0.9)
	scheduler = lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
	
	train_data = []
	# test_data = []
	sample_data = []
	for i in range(num_classes_in_task*task_id, num_classes_in_task*(task_id+1)):
		# task_data_i = MNIST_dataset.get_train_by_task(mode, task_id = i)
		train_data += MNIST_dataset.get_train_by_task(mode, task_id = i)[0:num_train_per_class]
		test_data += MNIST_dataset.get_test_by_task(mode, task_id = i)[0:num_test_per_class]
	train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
	# train_loader_size = len(train_data)	

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
	output1 = output.cpu()
	output1 = output1.numpy()
	#print(output1)
	output1 = output1[:gwr_imgs_per_task]
	# print(output1)
	# input('....................')
	labels1 = labels.numpy()
	labels1 = labels1[:gwr_imgs_per_task]
	# print(labels)
	# input('.....')
	# print(output.shape)
	# print(output[1].shape)

	labels1 = np.floor(labels1/num_classes_in_task)
	#print('task label',labels1)
	# g1 = gwr(act_thr = 0.95, fir_thr = 0.1, random_state=None, max_size=5000)
	if(task_id == 0):
		graph_gwr1 = g1.train(output1, labels1, n_epochs= gwr_epochs)
	else:
		graph_gwr1 = g1.train(output1, labels1, n_epochs= gwr_epochs, warm_start = True)
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

	train_MultiNet(MultiNet, criterion, optimizer, scheduler, output, labels, task_id, classifierNet_epoches, num_classes_in_task)
	
	#testing
	# test_data_ = []
	# for i in range(0 , task_id + num_classes_in_task):
	# 	test_data_ += MNIST_dataset.get_test_by_task(mode, task_id = i)
	
test_loader = torch.utils.data.DataLoader(test_data, batch_size = test_batch_size, shuffle = True)
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
output_test1 = output_test.cpu()
output_test1 = output_test1.numpy()
labels_test1 = labels_test.numpy()

labels_test1 = np.floor(labels_test1/num_classes_in_task)

nodes_per_tasks = g1.nodes_per_task(number_of_tasks)
print('nodes_per_task : ', nodes_per_tasks)

acc_g,class_by_class_acc_g = g1.test(output_test1, labels_test1,number_of_tasks)
print('GWR classification overall accuracy : ', acc_g)
print('GWR class_by_class accuracy : ', class_by_class_acc_g)

pred_tasks = g1.choose_task(output_test1,number_of_tasks)
pred_tasks = torch.from_numpy(pred_tasks)

acc,class_by_class_acc = test_MultiNet(MultiNet, criterion, optimizer, scheduler, output_test, 
	labels_test, pred_tasks, number_of_tasks*num_classes_in_task, num_classes_in_task)
print('GWR Multihead classification overall accuracy : ', acc)
print('GWR Multihead class_by_class accuracy : ', class_by_class_acc)

	# print('approach 1 ......')
	# print('number of nodes: ',num_nodes1)
	# print('number of clusters: ',number_of_clusters1)
	# acc,class_by_class_acc = g1.test(output_test, labels_test)
	# print('gwr clustering overall accuracy : ', acc)
	# print('gwr clustering class_by_class accuracy : ', class_by_class_acc)
	# # np.savez('acc_file1.npz', acc, class_by_class_acc)

	# print('Using K-Nearest ..')
	# acc,class_by_class_acc = g1.KNearest_test(output_test, labels_test)
	# print('gwr clustering overall accuracy : ', acc)
	# print('gwr clustering class_by_class accuracy : ', class_by_class_acc)
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