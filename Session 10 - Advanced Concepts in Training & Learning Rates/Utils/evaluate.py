# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 17:44:29 2020

@author: Anjan 
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def show_misclassified_images(model, device, dataset, classes,number = 25):
  misclassified_images = []
  
  for images, labels in dataset:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(labels)):
              if(len(misclassified_images)<number and predicted[i]!=labels[i]):
                misclassified_images.append([images[i],predicted[i],labels[i]])
            if(len(misclassified_images)>number):
              break
  return misclassified_images

def plot_misclassified_images(misclassified_images, classes,number = 25) :

  fig = plt.figure(figsize = (8,8))
  for i in range(25):
        sub = fig.add_subplot(5, 5, i+1)
        img = misclassified_images[i][0].cpu()
        img = img *2  + 0.5 
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg,(1, 2, 0)),interpolation='none')

        sub.set_title("P={}, A={}".format(str(classes[misclassified_images[i][1].data.cpu().numpy()]),str(classes[misclassified_images[i][2].data.cpu().numpy()])))
        sub.axis('off')
  plt.tight_layout()

  
  
def evaluate_classwise_accuracy(model, device, classes, test_loader):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(c)):
            	label = labels[i]
            	class_correct[label] += c[i].item()
            	class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
