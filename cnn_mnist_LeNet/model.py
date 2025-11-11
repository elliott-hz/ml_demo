from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import categorical_crossentropy, MeanSquaredError

# Pre-defined optimizers (keep for user choice)
adam_optimizer = Adam(learning_rate=0.01)
sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9)
rmsprop_optimizer = RMSprop(learning_rate=0.01)

# Pre-defined loss functions
cross_entropy_loss = categorical_crossentropy
mse_loss = MeanSquaredError()


def build_cnn_model(input_shape=(28, 28, 1), num_classes=10, optimizer=None):
    """
    Build a classic LeNet-5 model.

    This implements the LeNet-5 topology (Y. LeCun et al.) with a small compatibility
    helper: if your input images are 28x28 (MNIST), the model pads them to 32x32
    so the original layer sizes line up. If you already feed 32x32 images, set
    `input_shape=(32,32,1)` and the initial padding will still work (it will add
    2 pixels of padding on each side resulting in 36x36 unless you change it).

    Args:
        input_shape: tuple, input image shape, default (28,28,1) for MNIST.
        num_classes: int, number of output classes (default 10).
        optimizer: Keras optimizer instance to compile the model with. If None,
                   defaults to the predefined `sgd_optimizer`.

    Returns:
        model: a compiled Keras Model implementing LeNet-5.
    """
    # Use provided optimizer or fall back to predefined SGD (classic choice)
    if optimizer is None:
        optimizer = sgd_optimizer

    # Build LeNet-5
    model = models.Sequential([
        # Pad 28x28 -> 32x32 so original LeNet receptive fields match MNIST
        layers.ZeroPadding2D(padding=2, input_shape=input_shape),

        # C1: convolution 6 maps, 5x5, activation tanh
        layers.Conv2D(6, (5, 5), activation='tanh', kernel_initializer='glorot_uniform'),
        # S2: subsampling (average pooling) 2x2
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),

        # C3: convolution 16 maps, 5x5
        layers.Conv2D(16, (5, 5), activation='tanh', kernel_initializer='glorot_uniform'),
        # S4: subsampling
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),

        # C5: convolution producing 120 feature maps (5x5 conv over 5x5 -> 1x1)
        layers.Conv2D(120, (5, 5), activation='tanh', kernel_initializer='glorot_uniform'),

        layers.Flatten(),
        # F6: fully-connected 84
        layers.Dense(84, activation='tanh', kernel_initializer='glorot_uniform'),
        # Output layer
        layers.Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform')
    ])

    model.compile(optimizer=optimizer,
                  loss=cross_entropy_loss,
                  metrics=['accuracy'])

    return model