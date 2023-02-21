# ASL Recognition System using OpenCV and ResNet

This is a deep learning computer vision project that uses OpenCV and ResNet to recognize American Sign Language (ASL) gestures in real-time using a computer's webcam or phone's camera. 

## Requirements
- Python 3.x
- OpenCV 4.x
- PyTorch 1.x

## Dataset
The dataset used to train the ResNet model was derived from the publicly available [ASL Alphabet dataset on Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet). 

## Preprocessing
Prior to training the ResNet model, the images in the dataset were preprocessed using the following transformations:
- Resizing to (64, 64)
- Converting to PyTorch tensors
- Normalizing using the mean and standard deviation of the ImageNet dataset

## Training
The ResNet model was trained on the preprocessed dataset using the following hyperparameters:
- Learning rate: 0.001
- Batch size: 64
- Number of epochs: 3

## ResNet | CNN models with Residual Blocks
ResNet uses residual blocks to address the problem of vanishing gradients in deep neural networks. The residual block consists of two or more convolutional layers, and each block includes a shortcut connection that allows the gradient to flow directly through the block. This makes it easier for the network to learn the identity function and helps to prevent the gradient from vanishing.

Our model has two convolutional layers with ReLU activation, and a shortcut connection that adds the input to the output after passing it through the convolutional layers. 