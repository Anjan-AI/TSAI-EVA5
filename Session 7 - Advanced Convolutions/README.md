# ___Assignment 7___
## By 
1. Anjan Kumar Patra : anjan.student@gmail.com
2. Pradipt Trivedi : Pradipt.trivedi@gmail.com
3. Avnish Midha  :  avnishbm@gmail.com
4. Ramkumar M :  rkm1415@gmail.com
## Model Summary 
-----------------------
```
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
              ReLU-2           [-1, 16, 32, 32]               0
       BatchNorm2d-3           [-1, 16, 32, 32]              32
           Dropout-4           [-1, 16, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           4,608
              ReLU-6           [-1, 32, 32, 32]               0
       BatchNorm2d-7           [-1, 32, 32, 32]              64
           Dropout-8           [-1, 32, 32, 32]               0
         MaxPool2d-9           [-1, 32, 16, 16]               0
           Conv2d-10           [-1, 64, 16, 16]          18,432
             ReLU-11           [-1, 64, 16, 16]               0
      BatchNorm2d-12           [-1, 64, 16, 16]             128
          Dropout-13           [-1, 64, 16, 16]               0
           Conv2d-14           [-1, 64, 16, 16]          36,864
             ReLU-15           [-1, 64, 16, 16]               0
      BatchNorm2d-16           [-1, 64, 16, 16]             128
          Dropout-17           [-1, 64, 16, 16]               0
        MaxPool2d-18             [-1, 64, 8, 8]               0
           Conv2d-19            [-1, 128, 8, 8]          73,728
             ReLU-20            [-1, 128, 8, 8]               0
      BatchNorm2d-21            [-1, 128, 8, 8]             256
          Dropout-22            [-1, 128, 8, 8]               0
           Conv2d-23            [-1, 128, 8, 8]         147,456
             ReLU-24            [-1, 128, 8, 8]               0
      BatchNorm2d-25            [-1, 128, 8, 8]             256
          Dropout-26            [-1, 128, 8, 8]               0
        MaxPool2d-27            [-1, 128, 4, 4]               0
           Conv2d-28            [-1, 128, 4, 4]         147,456
             ReLU-29            [-1, 128, 4, 4]               0
      BatchNorm2d-30            [-1, 128, 4, 4]             256
          Dropout-31            [-1, 128, 4, 4]               0
           Conv2d-32            [-1, 128, 4, 4]           1,152
           Conv2d-33            [-1, 256, 4, 4]          32,768
             ReLU-34            [-1, 256, 4, 4]               0
      BatchNorm2d-35            [-1, 256, 4, 4]             512
          Dropout-36            [-1, 256, 4, 4]               0
AdaptiveAvgPool2d-37            [-1, 256, 1, 1]               0
           Linear-38                   [-1, 10]           2,570
================================================================
Total params: 467,098
Trainable params: 467,098
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.31
Params size (MB): 1.78
Estimated Total Size (MB): 5.11
----------------------------------------------------------------
```
Here's the accuracy logs
------
```
  0%|          | 0/391 [00:00<?, ?it/s]EPOCH: 0
Loss=1.2670176029205322 Batch_id=390 Accuracy=47.83: 100%|██████████| 391/391 [00:17<00:00, 21.82it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0100, Accuracy: 5534/10000 (55.34%)

EPOCH: 1
Loss=0.8131823539733887 Batch_id=390 Accuracy=61.92: 100%|██████████| 391/391 [00:18<00:00, 21.12it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0071, Accuracy: 6849/10000 (68.49%)

EPOCH: 2
Loss=0.8417479395866394 Batch_id=390 Accuracy=67.61: 100%|██████████| 391/391 [00:18<00:00, 21.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0067, Accuracy: 7109/10000 (71.09%)

EPOCH: 3
Loss=0.978293240070343 Batch_id=390 Accuracy=70.84: 100%|██████████| 391/391 [00:17<00:00, 21.74it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0061, Accuracy: 7338/10000 (73.38%)

EPOCH: 4
Loss=0.7869402170181274 Batch_id=390 Accuracy=73.25: 100%|██████████| 391/391 [00:18<00:00, 21.26it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0058, Accuracy: 7533/10000 (75.33%)

EPOCH: 5
Loss=0.6825851202011108 Batch_id=390 Accuracy=75.18: 100%|██████████| 391/391 [00:18<00:00, 21.56it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0055, Accuracy: 7667/10000 (76.67%)

EPOCH: 6
Loss=0.8117278814315796 Batch_id=390 Accuracy=76.60: 100%|██████████| 391/391 [00:19<00:00, 20.40it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0053, Accuracy: 7682/10000 (76.82%)

EPOCH: 7
Loss=0.7736882567405701 Batch_id=390 Accuracy=77.59: 100%|██████████| 391/391 [00:19<00:00, 20.41it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0052, Accuracy: 7807/10000 (78.07%)

EPOCH: 8
Loss=1.0532768964767456 Batch_id=390 Accuracy=78.55: 100%|██████████| 391/391 [00:19<00:00, 20.47it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0050, Accuracy: 7883/10000 (78.83%)

EPOCH: 9
Loss=0.49845701456069946 Batch_id=390 Accuracy=79.36: 100%|██████████| 391/391 [00:19<00:00, 20.08it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0045, Accuracy: 8072/10000 (80.72%)

EPOCH: 10
Loss=0.5687121748924255 Batch_id=390 Accuracy=80.11: 100%|██████████| 391/391 [00:19<00:00, 20.20it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0047, Accuracy: 7966/10000 (79.66%)

EPOCH: 11
Loss=0.5451533198356628 Batch_id=390 Accuracy=80.91: 100%|██████████| 391/391 [00:18<00:00, 20.78it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0047, Accuracy: 8024/10000 (80.24%)

EPOCH: 12
Loss=0.6770356297492981 Batch_id=390 Accuracy=81.41: 100%|██████████| 391/391 [00:18<00:00, 20.90it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0045, Accuracy: 8075/10000 (80.75%)

EPOCH: 13
Loss=0.3159541189670563 Batch_id=390 Accuracy=82.06: 100%|██████████| 391/391 [00:19<00:00, 20.37it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0042, Accuracy: 8227/10000 (82.27%)

EPOCH: 14
Loss=0.5060826539993286 Batch_id=390 Accuracy=82.55: 100%|██████████| 391/391 [00:19<00:00, 20.32it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0041, Accuracy: 8253/10000 (82.53%)

EPOCH: 15
Loss=0.548211932182312 Batch_id=390 Accuracy=82.61: 100%|██████████| 391/391 [00:19<00:00, 20.18it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0039, Accuracy: 8294/10000 (82.94%)

EPOCH: 16
Loss=0.4341511130332947 Batch_id=390 Accuracy=83.34: 100%|██████████| 391/391 [00:19<00:00, 19.95it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0040, Accuracy: 8292/10000 (82.92%)

EPOCH: 17
Loss=0.5264232158660889 Batch_id=390 Accuracy=83.84: 100%|██████████| 391/391 [00:19<00:00, 19.96it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0040, Accuracy: 8286/10000 (82.86%)

EPOCH: 18
Loss=0.3640556037425995 Batch_id=390 Accuracy=83.85: 100%|██████████| 391/391 [00:19<00:00, 20.03it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0041, Accuracy: 8354/10000 (83.54%)

EPOCH: 19
Loss=0.3883877098560333 Batch_id=390 Accuracy=84.34: 100%|██████████| 391/391 [00:18<00:00, 20.79it/s]

Test set: Average loss: 0.0040, Accuracy: 8361/10000 (83.61%)

```
