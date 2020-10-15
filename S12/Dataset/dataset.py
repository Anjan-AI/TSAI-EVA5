import torch
import torchvision
import os
import requests
import zipfile
from tqdm import tqdm
from io import BytesIO
import glob
from PIL import Image
import numpy as np

def cifar10_classes():
	return (
		'plane', 'car', 'bird', 'cat', 'deer',
		'dog', 'frog', 'horse', 'ship', 'truck'
	)


class Dataset:

	def __init__(self, train_transforms, test_transforms):

		self.train_transforms = train_transforms
		self.test_transforms = test_transforms

	def download_cifar10dataset(self, train=False):

		if train:
			return torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=self.train_transforms)
		else:
			return torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=self.test_transforms)

	def data_loader(self, dataset, cuda=False, batch_size=128, num_workers=4):

		# dataloader arguments - something you'll fetch these from cmdprmt
		dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers,
							   pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

		# train dataloader
		self.dataset_loader = torch.utils.data.DataLoader(
			dataset, **dataloader_args)

		return self.dataset_loader


class TinyImageNet(torch.utils.data.Dataset):

	"""Tiny Imagenet from cs231n
	"""

	def __init__(self, url='http://cs231n.stanford.edu/tiny-imagenet-200.zip',path='./tiny-imagenet-200'):
		self.path = path
		self.url = url
		self.classes = []
		self.data = []
		self.target = []

		if not os.path.isdir(self.path):
			self.download_dataset()
		print('TinyImageNet Downloaded')

		with open (self.path+'/wnids.txt','r') as f:
			wnids = [l.strip() for l in f]
		print(f'Found {len(wnids)} classes')
		self.classes = wnids
		for wclass in tqdm(wnids, desc='Loading Training Data...'):
			for file in glob.glob(f'{self.path}/train/{wclass}/images/*.JPEG'):
				img = Image.open(file)
				img = np.asarray(img)
				self.data.append(img)
				# To get labels as simple numbers from 0 to len(classes)-1
				self.target.append(self.classes.index(wclass))

		with open(self.path+'/val/val_annotations.txt','r') as f:
			for line in tqdm(f, desc='Loading Validation Data...'):
				line = line.strip()
				img_file, img_class = line.split('\t')[:2]
				img = np.asarray(Image.open(f'{self.path}/val/{img_file}'))
				self.data.append(img)
				self.target.append(self.classes.index(img_class))
		

	def  __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		#idx is index
		img = self.data[idx]
		target = self.target[idx]
		return img, target

	def download_dataset(self):
		print('Downloading TinyImageNet')
		r = requests.get(self.url, stream=True)
		if r.status_code == 200:
			file_size = int(r.headers.get('content-length','0'))
			zip_ = BytesIO()
			chunk_size = 1024**2
			for data in tqdm(iterable = r.iter_content(chunk_size = chunk_size), total = file_size//chunk_size, unit = 'MB', desc='Downloading...'):
				zip_.write(data)
			zip_ = zipfile.ZipFile(zip_)
			for file in tqdm(zip_.namelist(), total=len(zip_.namelist()), desc='Extacting Zip...'):
				zip_.extract(file)
			zip_.close()
		else:
			print(f'Got status code {r.status_code} for url {self.url} ')


data = TinyImageNet()