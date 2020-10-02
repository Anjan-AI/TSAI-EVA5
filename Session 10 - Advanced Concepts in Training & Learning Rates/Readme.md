# EVA5 - WEEK 9 Assignment #

## Assignment: ##
1. Move your last code's transformations to Albumentations. Apply ToTensor, HorizontalFlip, Normalize (at min) + More (for additional points)
2. Please make sure that your test_transforms are simple and only using ToTensor and Normalize
3. Implement GradCam function as a module. 
4. Your final code (notebook file) must use imported functions to implement transformations and GradCam functionality
5. Target Accuracy is 87%
6. Submit answers to S9-Assignment-Solution. 

## Below are the Training Details  ##
### Parameters and Hyperparameters ###
1. Loss Function: Cross Entropy Loss 
2. Optimizer: SGD
3. Learning Rate: 0.01
4. LR Step Size: 25
5. LR Gamma: 0.1
6. Batch Size: 64
7. num_workers = 4 
8. Epochs: 33
9. Data Augmentation (horizontal_flip_prob=0.6,vertical_flip_prob=0.6,rotate_degree =30.0,cutout)

### The following data augmentation techniques were applied to the dataset during training: ###
Random Rotation: 30 degrees
Random Horizontal Flip
verticle Flip


### the accurcay of 87% was reached in the 37th  Epoch. ###

### Submitted By  ###
1. Avnish Midha 
2. Ramkumar M 
3. Pradipt Trivedi 
4. Anjan Kumar Patra


