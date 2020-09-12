import torch
from torchvision import datasets, transforms
import config

train_data = None
test_data = None
train_loader = None
test_loader = None

def LoadData(): 
    global train_data, test_data
    train_transforms = transforms.Compose([
                            transforms.RandomRotation((-10.0, 10.0), fill=(1,)),
                            # transforms.RandomErasing(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])

    test_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])

    train_data = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

def InitialiseDataLoaders(): 
    global train_loader, test_loader
    dataloader_args = dict(shuffle=True, batch_size=config.batch_size, num_workers=4, pin_memory=True) if config.use_cuda else dict(shuffle=True, batch_size=int(config.batch_size/2))

    # load the training data and perform standard normalization 
    # parameter for normalization is mean and std dev.
    # train dataloader
    train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)


