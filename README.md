# AlexNet Experiments
This is project is a PyTorch implementation of the AlexNet architeture described in [ImageNet Classification with Deep Convolutional Neural Networks]. The following experiments have been performed on the architecture:

- Activation Unit: ReLU vs Tanh
- Dropout v/s No Dropout
- Overlapping Pooling v/s Non-Overlapping Pooling
- Optimization Techniques: SGD v/s SGD with momentum v/s SGD with momentum and weight decay v/s ADAM.

The experiments were done on a 35 class subset of the ImageNet Dataset that can be downloaded from [here](https://drive.google.com/file/d/0BzGkOSRHmXLrYjAybXljeDRfak0/view). The results of all the experiments can are compiled in [results.pdf](https://github.com/Shivanshu-Gupta/AlexNet-Experiments/blob/master/Results.pdf). The best model that attained over 76% accuracy on the the test-set can be downloaded from [here](https://drive.google.com/open?id=0By07sE0zY59RRnhVd1VjUURlSWs).

# Requirements
- python 3.5
- [pytorch]

# Usage
Use the following to get help with usage.
```sh
python main.py --help
```

[ImageNet Classification with Deep Convolutional Neural Networks]: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
[pytorch]: http://pytorch.org/
