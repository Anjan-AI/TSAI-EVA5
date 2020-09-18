import torchvision.transforms as transforms

def data_transforms(augmentation = False ,mean=(0.5,0.5,0.5) ,std_dev =(0.5,0.5,0.5) ,rotation =0.0, resize_scale =(0.08,1.0) ):
    
    trasform_lsit = [transforms.ToTensor(),
                     transforms.Normalize((mean), (std_dev))]
    
    if augmentation:
       trasform_lsit = [
            # Rotate image by 6 degrees
            transforms.RandomRotation((-rotation, rotation)),
            transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
        ] + trasform_lsit 
       
    return transforms.Compose(trasform_lsit)
