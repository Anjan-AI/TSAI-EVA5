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
#from Dataset.image_augmentations import AlbumentationTransforms
from Dataset.image_aug import AlbumentationTransforms
from Utils.SetCuda import set_seed,initialize_cuda
from Utils.utilities import print_model_summary,cross_entropy_loss
from Utils.evaluate import show_misclassified_images,evaluate_classwise_accuracy,plot_misclassified_images
from Utils.GradCam import GradCamView
from Utils.show_image import imshow ,show_train_data
#from models.resnet import ResNet18
from models.S11_model import NewResnet
from Training.Training_Testing import train ,test,runmodel
from Utils.plot import plot_metric
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from Utils.lr_finder import LRFinder
import albumentations as A
from Utils.LRtest import LR_test
classes = cifar10_classes()

channel_means = (0.4914, 0.4822, 0.4465)
channel_stdevs = (0.2471, 0.2435, 0.2616)
train_transform = AlbumentationTransforms([
                                       A.Rotate((-30.0, 30.0)),
                                       A.HorizontalFlip(),
                                       A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
                                       A.Normalize(mean=channel_means, std=channel_stdevs),
                                       A.Cutout(num_holes=2, max_h_size=9,max_w_size = 9,p=1) 
                                       ])
test_transform = AlbumentationTransforms([A.Normalize(mean=channel_means, std=channel_stdevs)])


Data = Dataset(train_transform,test_transform)



train_set = Data.download_cifar10dataset(train = True)
test_set = Data.download_cifar10dataset(train = False)



seed =1
cuda,device = initialize_cuda(seed)

train_loader = Data.data_loader(train_set,cuda= cuda ,batch_size = 32 , num_workers = 4 )
test_loader = Data.data_loader(test_set,cuda= cuda ,batch_size = 32 , num_workers = 4 )

show_train_data(train_loader, classes)

model = NewResnet()
print_model_summary(model, input_size = (3,32,32),device =device)




EPOCHS = 3
train_losses = []
test_losses = []
train_acc = []
test_acc = []


criterion = cross_entropy_loss()  # Create loss function
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
#scheduler = StepLR(optimizer, step_size=6, gamma=0.1) 
lr_finder = LRFinder(model, optimizer, criterion, device=device)
lr_finder.range_test(train_loader, val_loader=test_loader, end_lr=10, num_iter=5 ,step_mode="exp")

lr_finder.plot()



train_losses = []
test_losses = []
train_acc = []
test_acc = []

runmodel(model,device,train_loader,test_loader,optimizer,scheduler,EPOCHS,criterion, train_losses,train_acc,test_losses,test_acc)




