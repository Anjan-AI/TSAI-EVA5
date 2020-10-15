# EVA5 - WEEK 11 Assignment : Super Convergence #

## Assignment: ##
1. Write a code whichuses this new ResNet Architecture (Used in Dawn Bench) for Cifar10:
	PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
	Layer1 -
		X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
		R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
	Add(X, R1)
	Layer 2 -
		Conv 3x3 [256k]
		MaxPooling2D
		BN
		ReLU
	Layer 3 -
		X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
		R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
		Add(X, R2)
	MaxPooling with Kernel Size 4
	FC Layer 
	SoftMax 

2. Uses One Cycle Policy such that:
	Total Epochs = 24
	Max at Epoch = 5
	LRMIN = FIND
	LRMAX = FIND
	NO Annihilation

3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
4. Batch size = 512
5. Target Accuracy: 90%

## Below are the Training Details  ##
### Parameters and Hyperparameters ###
1. Loss Function: Cross Entropy Loss 
2. Optimizer: SGD
3. Batch Size: 512
4. num_workers = 4 
4. momentum=0.9
5. useed LR finder with end_lr=0.02 and step_mode="linear" ( could be Exp for bigger exp)
6. Best LR found  = 0.019866510851419033
7. Epochs: 25
9. Data Augmentation : As mentioned in the assignment above
10.Used OneCycleLR from pytorch API as the scheduler . [ Note : this Scheduler updated every batch , so change in training code , look at the train function inside training API]
11. max_lr = Best_lr
12. div_factor=25, final_div_factor=1


### Traing and test Accuracy 
Best Training Accuracy : 91.25%
Best Test Accuracy : 88.35%

Please refere to the code for Gradcam and Accuracy plots. 




