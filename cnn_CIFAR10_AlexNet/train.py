import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


def train(model, train_ds, val_ds, epochs=20, batch_size=128, model_path='cnn_CIFAR10_AlexNet/results/small_alexnet.h5'):
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    # ensure dir exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    return model, history

