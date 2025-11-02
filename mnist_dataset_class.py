from __future__ import print_function, division
import torch
import os
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

#################################################
	# Permutation Generator
#################################################
def generate_mnist_permutation(seed): 
	np.random.seed(seed)
	h = w = 28
	perm_inds = list(range(h*w))
	np.random.shuffle(perm_inds)
	return perm_inds

#################################################
	# MNIST Dataset Class
#################################################
class MNIST(Dataset):
	def __init__(self, class_by_class_dir, train_dir, test_dir):
		self.class_by_class_dir = class_by_class_dir
		self.train_dir = train_dir
		self.test_dir = test_dir

	def get_train_set(self, idx = None):
		sample = torch.load(self.train_dir)
		return sample

	def get_test_set(self, idx = None):
		sample = torch.load(self.test_dir)
		return sample

	def get_train_by_task(self, mode, task_id = None, permutation_idx = None):
		if mode == 'incremental_nc':
			sample = torch.load(os.path.join(self.class_by_class_dir, 'train', 'class_' + str(task_id) + '.pt'))
			return sample
		elif mode == 'incremental_ni':
			sample = []
			data = torch.load(self.train_dir)
			for idx in range(len(data)):
				sample.append([data[idx][0].float().view(-1)[permutation_idx].view(1,28,28), data[idx][1]])
			return sample

	def get_test_by_task(self, mode, task_id = None, permutation_idx = None):
		if mode == 'incremental_nc':
			sample = torch.load(os.path.join(self.class_by_class_dir, 'test', 'class_' + str(task_id) + '.pt'))
			return sample
		elif mode == 'incremental_ni':
			sample = []
			data = torch.load(self.test_dir)
			for idx in range(len(data)):
				sample.append([data[idx][0].float().view(-1)[permutation_idx].view(1,28,28), data[idx][1]])
			return sample			

