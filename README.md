# cnn2d
A work-in-progress vanilla implementation of a `convolutional autoencoder neural network` written in C, developed as an external object for the `Pure Data`.

The project aims to integrate machine learning algorithms into the `Pure Data` environment and to provide a configurable neural network architecture (with no external machine learning dependencies) designed for technical studies and real-time composition applications.


# Key Features
✅ Full autoencoder CNN implementation in plain C — no external libraries used

🧩 Dynamically creates array-based matrices for each layer:

``input``, ``kernels``, ``convolution``, ``pooling``, etc

🧠 Kernel initialization using ``He`` or ``Xavier`` methods

🎯 Bias values are initialized with small random values close to zero


⚙️ Supports optimization using:

``SGD``

``Adam``


🔧 Configurable components:

``Activation function``

``Loss function``

``Pooling method``


📐 Automatically calculates matrix dimensions based on:

``Number of layers``

``Input size``

``Kernel size``

``Padding``

``Pooling``

``Stride``


🚀 The model supports:

``Training``

``Evaluation``

``Input reconstruction``

``Latent space visualization``

📉 Displays reconstruction error during training




# Build
> [!NOTE]
`cnn2d` uses `pd.build`. To build the external on Linux, Mac, and Windows (using Mingw64):

1. `git clone https://github.com/oviniciuscesar/cnn2d/ --recursive`;
2. `cd cnn2d`;
4. `cmake . -B build`;
5. `cmake --build build`;
