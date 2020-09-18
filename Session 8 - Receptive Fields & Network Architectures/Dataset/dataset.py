import torch
import torchvision

def cifar10_classes():
    return (
        'plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    )

class Dataset:

  def __init__(self,train_transforms,test_transforms):
    
    self.train_transforms = train_transforms
    self.test_transforms = test_transforms

  def download_cifar10dataset(self, train=False):
      
    if train:
        return torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=self.train_transforms)
    else:
        return torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=self.test_transforms)


  def data_loader(self, dataset,cuda= False ,batch_size = 128 , num_workers = 4 ):
    
    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)


    # train dataloader
    self.dataset_loader = torch.utils.data.DataLoader(dataset, **dataloader_args)

    return self.dataset_loader

  
