This code can be used to reproduce results from our paper [Bayesian tensorized neural networks with automatic rank selection](https://www.sciencedirect.com/science/article/pii/S0925231221006950). 

We have since improved our method, and recommend that you use the faster Stochastic Variational Inference approach detailed in our paper [Towards Compact Neural Networks via End-to-End Training: A Bayesian Tensor Approach with Automatic Rank Determination
](https://arxiv.org/abs/2010.08689) with code available at https://github.com/colehawkins/bayesian-tensor-rank-determination.


## Setup

This code was run using Tensorflow=1.13/4 and Tensorflow-Probability=0.8. Those are the primary dependencies. Since Tensorflow-Probability 0.8 is no longer available on pypi setting up dependencies will be challenging, but you can install from source. 

## Running the code

The MNIST is example is located in `mnist.py`. CIFAR-10 and CIFAR-100 VGG-style examples are located in `cifar10_conv.py` and `cifar_100_conv.py`. The ResNet example is locatedd in `tensor_resnet_cifar.py`.


## Questions? 
Feel free to raise an issue or email colepshawkins@gmail.com
