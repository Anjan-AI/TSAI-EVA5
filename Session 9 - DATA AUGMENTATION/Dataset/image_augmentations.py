#import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch  import ToTensor
import numpy as np
"""
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
"""
class AlbumentationTransforms:
    def __init__(self,augmentation = False ,mean=(0.5,0.5,0.5) ,std_dev =(0.5,0.5,0.5) ,horizontal_flip_prob=0,vertical_flip_prob=0,gaussian_blur_prob=0,rotate_degree=0,HueSaturationValue=0,cutout=False):
        transforms_list = []
        if augmentation:
            if horizontal_flip_prob > 0:  # Horizontal Flip
                transforms_list += [A.HorizontalFlip(p=horizontal_flip_prob)]
            if vertical_flip_prob > 0:  # Vertical Flip
                transforms_list += [A.VerticalFlip(p=vertical_flip_prob)]
            if gaussian_blur_prob > 0:  # Patch Gaussian Augmentation
                transforms_list += [A.GaussianBlur(p=gaussian_blur_prob)]  
            if rotate_degree > 0:  # Rotate image
                transforms_list += [A.Rotate(limit=rotate_degree)] 
            if HueSaturationValue > 0:  # Rotate image
                transforms_list += [A.HueSaturationValue(p=0.3)]
            if cutout:
                transforms_list += [A.Cutout(num_holes=1, max_h_size=16,max_w_size = 16,p=1)]
                        
        transforms_list += [
                     A.Normalize((mean), (std_dev)),
                     ToTensor()]
        self.transforms = A.Compose(transforms_list)
    
    def __call__(self, img):
        img = np.array(img)
        #print(img)
        return self.transforms(image=img)['image']
    


