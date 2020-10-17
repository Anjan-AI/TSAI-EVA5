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
from Dataset.dataset import GetTinyImageNet ## new
#from Dataset.image_augmentations import AlbumentationTransforms
from Dataset.image_aug import AlbumentationTransforms
from Utils.SetCuda import set_seed,initialize_cuda
from Utils.utilities import print_model_summary,cross_entropy_loss
from Utils.evaluate import show_misclassified_images,evaluate_classwise_accuracy,plot_misclassified_images
from Utils.GradCam import GradCamView
from Utils.show_image import imshow ,show_train_data
from Utils.show_image import show_train_data_imagenet
from models.resnet import ResNet18
#from models.S11_model import NewResnet
from Training.Training_Testing import train ,test,runmodel
from Utils.plot import plot_metric
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from Utils.lr_finder import LRFinder
import albumentations as A
from Utils.CyclicLRtest import LRRangeFinder
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


Data = GetTinyImageNet(train_transform, test_transform, split=0.7)

train_set = Data.get_dataset(train = True)
test_set = Data.get_dataset(train = False)
classes = Data.classes

len(train_set), len(test_set), len(classes)

#check for the GUP avaliblbity and manual seeding
seed =1
cuda,device = initialize_cuda(seed)
# Load the test and train data , set the batch size & Num_workeres.
train_loader = Data.data_loader(train_set,cuda= cuda ,batch_size = 64 , num_workers = 4 )
test_loader = Data.data_loader(test_set,cuda= cuda ,batch_size = 64 , num_workers = 4 )

show_train_data_imagenet(train_loader, classes,3)

# laod the model and print the summary
model = ResNet18(num_classes=200)
type(model)
print_model_summary(model, input_size = (3,64,64), device=device)








