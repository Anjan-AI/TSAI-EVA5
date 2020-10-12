# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 18:02:46 2020

@author: Anjan Kumar Patra
"""
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

def train(model, device,optimizer, train_loader,criterion, train_losses, train_acc,l1_factor=0,scheduler = False):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  total = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = criterion(y_pred, target)
    if l1_factor > 0:  # Apply L1 regularization
            l1_criteria = nn.L1Loss(size_average=False)
            regularizer_loss = 0
            for parameter in model.parameters():
                regularizer_loss += l1_criteria(parameter, torch.zeros_like(parameter))
            loss += l1_factor * regularizer_loss
    
    train_losses.append(loss)
    # Backpropagation
    loss.backward()
    optimizer.step()
    if(scheduler):
      scheduler.step()
    
        # Update pbar-tqdm
    

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)   
    
   
    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader,testloss,test_losses,test_acc):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
           
            test_loss +=  testloss(outputs, target).item()  # sum up batch loss
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))

train_scheduler = False
def runmodel(model,device,trainloader,testloader,optimizer,scheduler,EPOCHS,criterion, train_losses,train_acc,test_losses,test_acc,l1_factor=0,batch_scheduler = False):
  #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  #scheduler = StepLR(optimizer, step_size=6, gamma=0.1)    
  for epoch in range(EPOCHS):
      print("EPOCH:", epoch)
      if(batch_scheduler):
        train_scheduler = scheduler
      train(model, device,optimizer, trainloader,criterion, train_losses,train_acc,train_scheduler)
      
      test(model, device, testloader,criterion,test_losses,test_acc)
      if(not batch_scheduler): 
        scheduler.step(test_losses[-1])
      
