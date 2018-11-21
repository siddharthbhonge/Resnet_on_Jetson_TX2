# Evaluation of Resnet on Nvidia Jetson TX2

<br />Keras implementation of Resnet using the on-board camera of TX2.The Signs dataset was used to train this model.However,this can be used as a generalized image clasiification model.

![alt text](https://github.com/siddharthbhonge/Resnet_on_Jetson_TX2/blob/master/resnet.png)

##### Demo Link:https://www.youtube.com/watch?v=UQUWNcQjsqg&t=1s

## Getting Started

The neural network was trained on Nvidia Titan X  GPU.This model was later used with nvidia Jetson TX2 Board.
Opencv was used to capture images .

## Prerequisites

1.Python 3.5 <br />
2.Tensorflow 1.5<br />
3.Keras <br />
4.Scikit Learn<br />
5.Open CV 3.4.1<br />


## Theory

The details of this ResNet-50 model are:

![alt text](https://github.com/siddharthbhonge/Resnet_on_Jetson_TX2/blob/master/residula_block.png)
    Zero-padding pads the input with a pad of (3,3)<br/>
    Stage 1:<br/>
        The 2D Convolution has 64 filters of shape (7,7) and uses a stride of (2,2). Its name is "conv1".<br/>
        BatchNorm is applied to the channels axis of the input.<br/>
        MaxPooling uses a (3,3) window and a (2,2) stride.<br/>
    Stage 2:<br/>
        The convolutional block uses three set of filters of size [64,64,256], "f" is 3, "s" is 1 and the block is "a".<br/>
        The 2 identity blocks use three set of filters of size [64,64,256], "f" is 3 and the blocks are "b" and "c".<br/>
    Stage 3:<br/>
        The convolutional block uses three set of filters of size [128,128,512], "f" is 3, "s" is 2 and the block is "a".<br/>
        The 3 identity blocks use three set of filters of size [128,128,512], "f" is 3 and the blocks are "b", "c" and "d".<br/>
    Stage 4:<br/>
        The convolutional block uses three set of filters of size [256, 256, 1024], "f" is 3, "s" is 2 and the block is "a".<br/>
        The 5 identity blocks use three set of filters of size [256, 256, 1024], "f" is 3 and the blocks are "b", "c", "d", "e" and "f".<br/>
    Stage 5:<br/>
        The convolutional block uses three set of filters of size [512, 512, 2048], "f" is 3, "s" is 2 and the block is "a".<br/>
        The 2 identity blocks use three set of filters of size [512, 512, 2048], "f" is 3 and the blocks are "b" and "c".<br/>
    The 2D Average Pooling uses a window of shape (2,2) and its name is "avg_pool".<br/>
    The flatten doesn't have any hyperparameters or name.<br/>
    The Fully Connected (Dense) layer reduces its input to the number of classes using a softmax activation. Its name should be 'fc' + str(classes).<br/>





## Running the code

##### 1.Directory Structure
```
---------------------------------
Resnet_on_Jetson
|-resnet_utils.py
|-resnet_50.py
|-resnet_small.py
|-test_on_cam.py
|-data-|
|       |-train
|       |-test
|-----------------------------------

```


##### 2.Implementation on Camera Jetson

|-The Resnet 50 Model is really huge.<br />
|-It made the jetson really slow as it cold not fit into he memory<br/>
|-We created another small model for the same in file resnet_small.py<br />

![alt text](https://github.com/siddharthbhonge/Face_Recognition_with_jetson_TX2/blob/master/memory.png)

<br />The facenet aims to minimize this triplet loss.<br />
Here f(A,P,N) stands for the embeddings of each of the input image<br />


##### 3.Results 
![alt text](https://github.com/siddharthbhonge/Face_Recognition_with_jetson_TX2/blob/master/result.png)


## Authors

* **Siddharth Bhonge** - *Parser /Model* - https://github.com/siddharthbhonge


## Acknowledgments

* Andrew Ng  | Deeplearning.ai<br />
*the ResNet algorithm due to He et al. (2015). The implementation here also took significant inspiration and follows the structure given in the github repository of Francois Chollet:
*Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun - Deep Residual Learning for Image Recognition (2015)
*Francois Chollet's github repository: https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

