#import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch  import ToTensor
import numpy as np

class AlbumentationTransforms:
    """
     class to create test and train transforms using Albumentations
    """
 
    def __init__(self, transforms_list=[]):
        transforms_list.append(ToTensor())
        self.transforms = A.Compose(transforms_list)
    
    def __call__(self, img):
        img = np.array(img)
        return self.transforms(image=img)['image']

        


  
    
    


