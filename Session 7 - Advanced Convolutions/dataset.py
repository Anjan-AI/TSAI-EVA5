import torch
import torchvision


SEED = 1

class Dataset:

  def __init__(self,train_transforms,test_transforms):
    
    self.train_transforms = train_transforms
    self.test_transforms = test_transforms

  def set_dataset(self, train=False):
      
    if train:
        return torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=self.train_transforms)
    else:
        return torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=self.test_transforms)


  def get_dataset(self, dataset):
    # checking CUDA
    self.cuda = torch.cuda.is_available()
    # For reproducibility
    torch.manual_seed(SEED)
    if self.cuda:        
      torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if self.cuda else dict(shuffle=True, batch_size=64)


    # train dataloader
    self.dataset_loader = torch.utils.data.DataLoader(dataset, **dataloader_args)

    return self.dataset_loader
