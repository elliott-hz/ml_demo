import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, GRU
from keras import optimizers
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import SimpleRNN
from sklearn.metrics import classification_report
import os

# ----------------------------------------------------
# 1. Load & preprocess data
# ----------------------------------------------------
vocabulary = 10000
max_len = 50

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocabulary)

x_train = pad_sequences(x_train, maxlen=max_len)
x_test  = pad_sequences(x_test,  maxlen=max_len)

# Validation split
x_valid = x_train[:5000]
y_valid = y_train[:5000]
x_train = x_train[5000:]
y_train = y_train[5000:]

print("Training set shape:", x_train.shape, y_train.shape)
print("Validation set shape:", x_valid.shape, y_valid.shape)
print("Testing set shape:", x_test.shape, y_test.shape)

# ----------------------------------------------------
# 2. Build model
# ----------------------------------------------------
embedding_dim = 64
state_dim = 32

# Case 1: SimpleRNN Layer: return_sequences=False
model = Sequential([
    Embedding(input_dim=vocabulary, output_dim=embedding_dim, input_shape=(max_len,)),
    GRU(units=state_dim, return_sequences=False),
    Dense(1, activation='sigmoid')
])

# Case 2: SimpleRNN Layer: return_sequences=True

model.summary()

# ----------------------------------------------------
# 3. Compile
# ----------------------------------------------------
model.compile(
    optimizer=optimizers.RMSprop(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ----------------------------------------------------
# 4. Callbacks：EarlyStopping + ReduceLROnPlateau
# ----------------------------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,      # reduce to 0.5 LR
    patience=3,       # perform reducing LR if no improvement for at least patience epoches
    min_lr=1e-6
)

# ----------------------------------------------------
# 5. Train
# ----------------------------------------------------
epochs = 50
history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=32,
    validation_data=(x_valid, y_valid),
    callbacks=[early_stop, lr_reducer]
)

# ----------------------------------------------------
# 6. Plot Loss Curves
# ----------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig('result/loss_curve.png')
plt.close()  # Close the figure to free memory

# ----------------------------------------------------
# 7. Plot Accuracy Curves
# ----------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig('result/accuracy_curve.png')
plt.close()  # Close the figure to free memory

# ----------------------------------------------------
# 8. Classification Report
# ----------------------------------------------------
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype("int32")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Save the model
model.save('result/simple_rnn_gru_model.h5')