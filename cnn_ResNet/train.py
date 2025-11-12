"""Training utilities for cnn_ResNet.
Functions:
- compile_and_train(model, train_ds, val_ds, epochs, save_path, callbacks)
"""
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks


def compile_and_train(model, train_ds, val_ds, epochs=3, save_path=None):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        callbacks=[callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)]
    )

    if save_path:
        try:
            model.save(save_path)
            print(f"Model saved to {save_path}")
        except Exception as e:
            print(f"Warning: failed to save model: {e}")

    return history
