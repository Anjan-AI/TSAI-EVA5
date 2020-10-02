# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:33:42 2020

@author: Anjan 
"""
import torch
import os
import torchvision
import torchvision.transforms as transfroms
from Dataset.dataset import Dataset,cifar10_classes
from Dataset.image_augmentations import AlbumentationTransforms
from Utils.SetCuda import set_seed,initialize_cuda
from Utils.utilities import print_model_summary,cross_entropy_loss
from Utils.show_image import imshow ,show_train_data
from models.resnet import ResNet18
from Training.Training_Testing import train ,test,runmodel

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

classes = cifar10_classes()

train_transform = AlbumentationTransforms(augmentation = True, mean=(0.4914, 0.4822, 0.4465) ,
                                          std_dev =(0.2471, 0.2435, 0.2616),
                                          horizontal_flip_prob=0.6,
                                          vertical_flip_prob=0.6,
                                          rotate_degree =30.0,
                                          HueSaturationValue = 0.6,
                                          cutout= True)

test_transform = AlbumentationTransforms(augmentation = True, mean=(0.4914, 0.4822, 0.4465) ,std_dev =(0.2471, 0.2435, 0.2616))

Data = Dataset(train_transform,test_transform)



train_set = Data.download_cifar10dataset(train = True)
test_set = Data.download_cifar10dataset(train = False)



seed =1
cuda,device = initialize_cuda(seed)

train_loader = Data.data_loader(train_set,cuda= cuda ,batch_size = 32 , num_workers = 4 )
test_loader = Data.data_loader(test_set,cuda= cuda ,batch_size = 32 , num_workers = 4 )

show_train_data(train_loader, classes)

model = ResNet18()
print_model_summary(model, input_size = (3,32,32),device =device)

EPOCHS = 1
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



