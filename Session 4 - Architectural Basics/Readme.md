# EVA5 - WEEK 4 Assignment #

### Design a netwrok to train MNIST dataset with below conditions : ###
1. 99.4% validation accuracy
2. Less than 20k Parameters 
3. Less than 20 Epochs
4. No fully connected layer

total no of Parameter: 18698.
### below are the technique used ###
1. 3 X 3 kernal Convolution.
2. Batch Normalization after every convlution.
3. Dropout at each bloack.
4. 1X1 kernal based convolution to reduce the no of channel.
5. Max pooling used once after reaching recpetive filed of 7 X7 
6. GAP used as the last layer.
7. 99.41 % Test accuracy achieved in 14th Epoch.
