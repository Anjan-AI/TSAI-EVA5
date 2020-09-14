
import torch
import torch.nn.functional as F

test_losses = []
test_acc = []


  # Function to test 
'''
  Args: 
  Model : created model to be used for training
  device : GPU or cpu
  test_laoded: data on which the testing has to be done
  

'''
def test(model, device, test_loader): #, losstype):
    global test_losses, test_acc
    model.eval() # seting up the model for evalaution.
    test_loss = 0 # setting the test loss to 0
    correct = 0 # countign the no of coorect classfication.
    with torch.no_grad(): # turn off gradients, since we are in test mode
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)   # copy the data to device.
            output = model(data) # predict the output

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset) # calculating the test loss.
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Test Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc.append(100. * correct / len(test_loader.dataset))
    return test_losses, test_acc