# EVA5 - WEEK 1 Assignment #

### 1. What are Channels and Kernels (according to EVA)? ###
***ANS :***
**Kernel** are the matrix of odd no size but w.r.t EVA its 3X3, which is convolved with the image to extract certain features. 
<br/>The values of the kernel and the symmetry is responsible for a particular type of feature. 
for example, the below 3X3 kernel can be used to produce a horizontal edge.
|-1|-1 |-1|
|:---:|:---:|:---:|
| 0 |0 |0 |
| 1 | 1 |1 |

<br/> Output of the convolution of an image with a kernal is a channel, containing only a specific feature. As it extracts features,
<br/> on that pretext its also called a feature extractor. Feature to be extracted have not to be only vertical or horizontal edges, it can
<br/> be an arch, a blob, a  blank space etc.

__**Channels :**__
<br/> Channels are the outcome of images convolved with a definite kernel. each Channel contains only a specific feature i.e. a channel is a collection of same/similar features. For example, after convoluting an image with different kernals one at a time we will get as many channels and lets say one channel will have all the vertical 
edges present in image while other can have all the horizontal edges and similarly one specific feature contained in other channels.
<br/> In a more general context, in the below Image we can see that it is a Image with different Alphabets present in it.
<p align ="center">
  <img widht= 200, height = 200 src="Resources/Alphabet.PNG">			  
</p>

now in this image we can consider each alphabet as a feature. So each alphabet information can be stored in one channel. hence this image can be represented in 52 channels each representing a feature of small and capital letter in it. So we can think of our image as a set of different channels.

### 2. Why should we (nearly) always use 3x3 kernels? ###
***ANS :*** 
below are the reasons for mostly using 3X3 kernel
1. Using an odd size kernel makes the kernel symmetric. 
2. To get a receptive filed of 5X5 we can have one stage convolution with 5X5 kernel. 
<br/>to get the same receptive filed with 3X3 kernel we need to do 2 stages of convolution.
<br/> Image(5X5) -> Convolved with 5X5 -> 1X1
<br/> Image(5X5) -->Convolved with 3X3 --> 3X3 ->convolved with 3X3 -> 1X1
<p align ="center">
  <img  src="Resources/Kernel.png">			  
</p>
3. Number of multiplication operations while convolution using 3X3 kernal to replicate the effect of a bigger size kernal is lesser in comparision to bigger kernals.
<br/> If image size is 1004X1004, Continuing on the example shown in point 2
<br/> Number of Multiplication operations while using 5X5 kernal -> 1000 * 1000 * 25 -> 25000000 operations.
<br/> Number of Multiplication operations while using 3X3 kernal -> 1002 * 1002 * 9 + 1000 * 1000 * 9 -> 18036036 operations.

4. In recent time there are hardware’s which are accelerated for 3X3 kernel operations. 


### 3. How many times to we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...) ###
***ANS :***
99 Times 
<br/>
 199x199 > 197x197 > 195x195 > 193x193 > 191x191 > 189x189 > 187x187 > 185x185 > 183x183 > 181x181 > 179x179 > 177x177 > 175x175 > 173x173 > 171x171 > 169x169 > 167x167 > 165x165 > 163x163 > 161x161 > 159x159 > 157x157 > 155x155 > 153x153 > 151x151 > 149x149 > 147x147 > 145x145 > 143x143 > 141x141 > 139x139 > 137x137 > 135x135 > 133x133 > 131x131 > 129x129 > 127x127 > 125x125 > 123x123 > 121x121 > 119x119 > 117x117 > 115x115 > 113x113 > 111x111 > 109x109 > 107x107 > 105x105 > 103x103 > 101x101 > 99x99 > 97x97 > 95x95 > 93x93 > 91x91 > 89x89 > 87x87 > 85x85 > 83x83 > 81x81 > 79x79 > 77x77 > 75x75 > 73x73 > 71x71 > 69x69 > 67x67 > 65x65 > 63x63 > 61x61 > 59x59 > 57x57 > 55x55 > 53x53 > 51x51 > 49x49 > 47x47 > 45x45 > 43x43 > 41x41 > 39x39 > 37x37 > 35x35 > 33x33 > 31x31 > 29x29 > 27x27 > 25x25 > 23x23 > 21x21 > 19x19 > 17x17 > 15x15 > 13x13 > 11x11 > 9x9 > 7x7 > 5x5 > 3x3 > 1x1

### 4. How are kernels initialized? ###
***ANS :***
Initial weights of the kernel can be initialized using random values, though it is better to use more complex methods  like Xavier and MSRA initializations whose values may seem random or arbitrary but they help in faster convergence of the network. 
So if your model doesnt converge, you might want to look at your initialization strategies.

### 5.What happens during the training of a DNN? ###
***ANS :***
During training, back propagation is used to train the weights, from output towards the input layers.

Weights (in any layer ) are increased or decreased by looking into the gradient of the succeeding layer by the optimizer. 

This change is proportional to the learning rate (we set) and also the position of the layer from the output layer(ie. gradient of the previous layer).

<p align ="center">
  <img widht= 400, height = 400 src="Resources/errorback.png">			  
</p>

### Authors :###
1. Anjan Kumar Patra
2. Pradipt Trivedi
3. Ram Kumar
4.Avnish 
