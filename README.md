# VPN: Video Pixel Networks in Tensorflow 
The VPN is a probabilistic video model that estimates the discrete joint distribution of the raw pixel values in a video. It approaches the best possible performance on the Moving MNIST benchmark. 

This repository contains a tensorflow implementation of the VPN architecture proposed in the [paper](https://arxiv.org/abs/1610.00527). However, this code hasn’t been trained and tested on the full Moving MNIST dataset because of the lack of the computation power. It has been overfitted on one sequence to insure the correctness of the implementation. 

This repository also contains some additional experiments with the VPN architecture that are not mentioned in the original paper. These experiments are:
* Mini VPN architecture.
* Micro VPN architecture.
* Prediction-Based VPN architecture (ongoing work).
* Action conditioned VPN (ongoing work).

### The Moving MNIST Dataset
You can download the full moving MNIST dataset from toronto [website](http://www.cs.toronto.edu/~nitish/unsupervised_video/). 

### Training
```
python vpn.py
```
