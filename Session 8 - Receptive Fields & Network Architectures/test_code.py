# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:33:42 2020

@author: 20115260
"""
import torch
import os
import torchvision
import torchvision.transforms as transfroms
from Dataset.dataset import Dataset,cifar10_classes
from Dataset.image_augmentations import data_transforms
from Utils.SetCuda import set_seed,initialize_cuda
from Utils.utilities import print_model_summary,cross_entropy_loss
from models.resnet import ResNet18
from Training.Training_Testing import train ,test,runmodel

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

classes = cifar10_classes()

train_transforms = data_transforms(augmentation = True, mean=(0.4914, 0.4822, 0.4465) ,std_dev =(0.2471, 0.2435, 0.2616),rotation =12.0)
test_transforms = data_transforms()

Data = Dataset(train_transforms, test_transforms)

train_set = Data.download_cifar10dataset(train = True)
test_set = Data.download_cifar10dataset(train = True)

seed =1
cuda,device = initialize_cuda(seed)

train_loader = Data.data_loader(train_set,cuda= cuda ,batch_size = 128 , num_workers = 4 )
test_loader = Data.data_loader(test_set,cuda= cuda ,batch_size = 128 , num_workers = 4 )

model = ResNet18()
print_model_summary(model, input_size = (3,32,32),device =device)

EPOCHS = 3
train_losses = []
test_losses = []
train_acc = []
test_acc = []


criterion = cross_entropy_loss()  # Create loss function
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = StepLR(optimizer, step_size=6, gamma=0.1) 
train_losses = []
test_losses = []
train_acc = []
test_acc = []

runmodel(model,device,train_loader,test_loader,optimizer,scheduler,EPOCHS,criterion, train_losses,train_acc,test_losses,test_acc)





