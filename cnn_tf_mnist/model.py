from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import categorical_crossentropy, MeanSquaredError

# Pre-defined optimizers
adam_optimizer = Adam(learning_rate=0.01)
sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9)
rmsprop_optimizer = RMSprop(learning_rate=0.01)

# Pre-defined loss functions
cross_entropy_loss = categorical_crossentropy
mse_loss = MeanSquaredError()


def build_cnn_model():
    """
    Build CNN model with pre-defined optimizer and loss function
    
    Returns:
        model: Compiled Keras model
    """
    # Build the CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), kernel_initializer='he_uniform'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'),
        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_initializer='he_uniform'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax', kernel_initializer='he_uniform')
    ])
    
    # Compile the model with pre-defined optimizer and loss
    model.compile(optimizer=adam_optimizer,
                  loss=cross_entropy_loss,
                  metrics=['accuracy'])
    
    return model