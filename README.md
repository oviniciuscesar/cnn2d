# cnn2d
This repository contains a vanilla implementation of a convolutional autoencoder neural network written in C, developed as an external object for the Pure Data (Pd) environment.

The project aims to provide a lightweight, configurable neural network architecture designed for creative and real-time applications with no external machine learning dependencies.

✨ Key Features
Full autoencoder CNN implementation in plain C, without relying on external libraries;

Dynamic creation of array-based matrices for each layer of the network:

Input, kernels, convolution, pooling, and others;

Kernels are initialized using He or Xavier initialization methods;

Biases are initialized with small random values near zero;

Supports optimization using SGD or Adam;

Selectable components:

Activation function;

Loss function;

Pooling method;

Automatic calculation of matrix dimensions based on:

Number of layers, input size, kernel size, padding, pooling, and stride;

All matrices and helper vectors are dynamically allocated and freed as needed;

Vector arrays that store layer parameters include:

input_padded, convolv_matriz_size, pooling_matriz_size, kernels_size, pooling_size, stride_conv, stride_pool;

The model supports:

Training, evaluation, input reconstruction, and latent space display;

Error computation and display during training;

Tested with synthetic data, demonstrating learning and input reconstruction capabilities.


# Build
> [!NOTE]
`cnn2d` uses `pd.build`. To build the external on Linux, Mac, and Windows (using Mingw64):

1. `git clone https://github.com/oviniciuscesar/cnn2d/ --recursive`;
2. `cd cnn2d`;
4. `cmake . -B build`;
5. `cmake --build build`;
