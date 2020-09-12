
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from regularization import L1_loss

train_losses = []
train_acc = []

  # Function to train 
'''
  Args: 
  Model : created model to be used for training
  device : GPU or cpu
  train_laoded: data on which the training has to be done
  Optimizer : the optimization algorithm to be used
  epoch : no fo epoch 

'''
def train(model, device, train_loader, optimizer, epoch, losstype):
    global train_losses, train_acc
    model.train() # Set the model on training mode
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
      data, target = data.to(device), target.to(device) # moving the data to device
      optimizer.zero_grad() # zero the graidents 
      output = model(data) # getting the model output

      loss = 0
      if (losstype == "nll") or (losstype == "L2"):
        loss = F.nll_loss(output, target) # calculating the The negative log likelihood loss
      elif (losstype == "L1") or (losstype == "L1L2"):
        loss = L1_loss(device, output, target, model)

      train_losses.append(loss)
      loss.backward() # flowing the gradients backward.
      optimizer.step() # paameter updated basd on the current gradient.
      
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()
      processed += len(data)        
      
      pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx} Train Accuracy={100*correct/processed:0.2f}%')
      train_acc.append(100*correct/processed)
    return train_losses, train_acc
