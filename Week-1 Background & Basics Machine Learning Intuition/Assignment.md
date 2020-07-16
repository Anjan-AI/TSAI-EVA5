# EVA5 - WEEK 1 Assignment #

### 1. What are Channels and Kernels (according to EVA)? ###
***ANS :***
**Kernal** are the matrix of odd no size (usually 3X3) whcih is convolved with the image to extract certain features. 
<br/>The values of the kernal and the symetry is responsible for a particular type of feature. 
for exalple the below 3X3 keranl can be used to produce a hroizontal edge.

|-1|-1 |-1|
|:---:|:---:|:---:|
| 0 |0 |0 |
| 1 | 1 |1 |

__**Channels :**__
<br/> Channels are the outcome of images convloved with a definite kernal. each Channel consist diffnenrt features. for example in the below Image we can see that it is a Image with differnt Alphabetes present in it.
<p align ="center">
  <img widht= 200, height = 200 src="Resources/Alphabet.PNG">			  
</p>

now in this we can consider each alphabet as a feaure. so each alphabet information can be stored in one channel. hence this image can be represnted in 52 channels each represeting a feature of small and captial letter in it. 


### 2. Why should we (nearly) always use 3x3 kernels? ###
***ANS :*** 
below are the resons for mostly using 3X3 kernal
1. Using a odd size kernal makes the kernal symetric. 
2. To get a receptive filed of 5X5 we can have one stage covlution with 5X5 kernal. 
<br/>to get the same receptive filed with 3X3 kernal we need to do 2 stages of convution.
<br/> Image(5X5) -> Convolved with 5X5 -> 1X1
<br/> Image(5X5) -->Convolved with 3X3 --> 3X3 ->convolved with 3X3 -> 1X1
<p align ="center">
  <img  src="Resources/Kernel.PNG">			  
</p>

3. In recent time there are hardwares which are accelared for 3X3 kernal oerations. 


### 3. How many times to we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...) ###
***ANS :***
99 Times 
<br/>
 199x199 > 197x197 > 195x195 > 193x193 > 191x191 > 189x189 > 187x187 > 185x185 > 183x183 > 181x181 > 179x179 > 177x177 > 175x175 > 173x173 > 171x171 > 169x169 > 167x167 > 165x165 > 163x163 > 161x161 > 159x159 > 157x157 > 155x155 > 153x153 > 151x151 > 149x149 > 147x147 > 145x145 > 143x143 > 141x141 > 139x139 > 137x137 > 135x135 > 133x133 > 131x131 > 129x129 > 127x127 > 125x125 > 123x123 > 121x121 > 119x119 > 117x117 > 115x115 > 113x113 > 111x111 > 109x109 > 107x107 > 105x105 > 103x103 > 101x101 > 99x99 > 97x97 > 95x95 > 93x93 > 91x91 > 89x89 > 87x87 > 85x85 > 83x83 > 81x81 > 79x79 > 77x77 > 75x75 > 73x73 > 71x71 > 69x69 > 67x67 > 65x65 > 63x63 > 61x61 > 59x59 > 57x57 > 55x55 > 53x53 > 51x51 > 49x49 > 47x47 > 45x45 > 43x43 > 41x41 > 39x39 > 37x37 > 35x35 > 33x33 > 31x31 > 29x29 > 27x27 > 25x25 > 23x23 > 21x21 > 19x19 > 17x17 > 15x15 > 13x13 > 11x11 > 9x9 > 7x7 > 5x5 > 3x3 > 1x1

### 4. How are kernels initialized? ###
***ANS :*** 
<center>

 Centered text
 ------------
</center>

 Example text
 ============

Centered text
============