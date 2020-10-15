import torch 
import torch.nn as nn
import torch.optim as optim

from torchsummary import summary
from torch.optim.lr_scheduler import StepLR


def print_model_summary(model, input_size,device =torch.device("cpu")):
    
    print(device)
    model = model.to(device)
    print(summary(model, input_size=input_size))


def cross_entropy_loss():
    """Create Cross Entropy Loss
    Returns:
        Cross entroy loss function
    """
    return nn.CrossEntropyLoss()


def sgd_optimizer(model, learning_rate, momentum, l2_factor=0.0):
    """Create optimizer.
    Args:
        model: Model instance.
        learning_rate: Learning rate for the optimizer.
        momentum: Momentum of optimizer.
        l2_factor: Factor for L2 regularization.
    
    Returns:
        SGD optimizer.
    """
    return optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=l2_factor
    )


def lr_scheduler(optimizer, step_size, gamma):
    """Create LR scheduler.
    Args:
        optimizer: Model optimizer.
        step_size: Frequency for changing learning rate.
        gamma: Factor for changing learning rate.
    
    Returns:
        StepLR: Learning rate scheduler.
    """

    return StepLR(optimizer, step_size=step_size, gamma=gamma)