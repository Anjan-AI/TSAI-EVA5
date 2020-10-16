# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 18:46:51 2020

@author: Anjan 
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


# functions to show an image
# functions to show an image
def imshow(img,c ):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    fig = plt.figure(figsize=(7,7))
    plt.imshow(np.transpose(npimg, (1, 2, 0)),interpolation='none')
    plt.title(c)



def show_train_data(dataset, classes):

	# get some random training images

  dataiter = iter(dataset)
  images, labels = dataiter.next()
  for i in range(10):
    index = [j for j in range(len(labels)) if labels[j] == i]
    imshow(torchvision.utils.make_grid(images[index[0:5]],nrow=5,padding=2,scale_each=True),classes[i])

def imshow1(img, title):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    fig = plt.figure(figsize=(7,7))
    plt.title(title)
    plt.imshow(np.transpose(npimg, (1, 2, 0)),interpolation='none')
    
    

def show_train_data_imagenet(data_loader, classes, n_rows):
    # get some random training images
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    labels = [classes[label] for label in labels]
    for i in range(n_rows):
        title = '    '.join(labels[i*5:i*5+5])
        imshow1(torchvision.utils.make_grid(images[i*5:i*5+5],nrow=5,padding=2,scale_each=True), title)    