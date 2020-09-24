# FBSNNs_pytorch
A pytorch version of the FBSNNs (in tensorflow 1*) original created by Maziar Raissi. This is only used for study and learning purpose. 

To solve the Black-Scholes-Barenblatt problem, a deep neural network is built to track the evolutions of forward-backward stochastic differential equations. The auto-differetiation of neural network helps to approximate the feedback function (decsions). The most exciting part is that the size of the network is independent of the time horizon. 

For more detials, please refer to (https://maziarraissi.github.io/FBSNNs/) and (https://arxiv.org/abs/1804.07010) for the original pioneer work. 
