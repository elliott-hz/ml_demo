import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------------------------------------
# 1. Reproducibility（随机数固定）
# ----------------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)

# ----------------------------------------------------
# 2. Load & preprocess data
# ----------------------------------------------------
vocab_size = 10000
max_len = 100   # modern: 更长序列通常有更好的表达能力

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# pad to same length
x_train = pad_sequences(x_train, maxlen=max_len)
x_test  = pad_sequences(x_test,  maxlen=max_len)

# validation split
x_train, x_valid = x_train[5000:], x_train[:5000]
y_train, y_valid = y_train[5000:], y_train[:5000]

print("Train:", x_train.shape)
print("Valid:", x_valid.shape)
print("Test:", x_test.shape)

# ----------------------------------------------------
# 3. Build modern SimpleRNN model (Functional API)
# ----------------------------------------------------
embedding_dim = 64
rnn_units = 64

inputs = layers.Input(shape=(max_len,))

x = layers.Embedding(vocab_size, embedding_dim)(inputs)

x = layers.LSTM(
        rnn_units,
        dropout=0.2,
        recurrent_dropout=0.2
    )(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)
model.summary()

# ----------------------------------------------------
# 4. Compile
# ----------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------------------------------
# 5. Modern callbacks
# ----------------------------------------------------
if not os.path.exists("result"):
    os.makedirs("result")

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        filepath="result/best_lstm_rnn.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

# ----------------------------------------------------
# 6. Train
# ----------------------------------------------------
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(x_valid, y_valid),
    callbacks=callbacks
)

# ----------------------------------------------------
# 7. Plot curves
# ----------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train", "val"])
plt.grid()
plt.savefig("result/loss_lstm.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["train", "val"])
plt.grid()
plt.savefig("result/accuracy_lstm.png")
plt.close()

# ----------------------------------------------------
# 8. Evaluation
# ----------------------------------------------------
y_pred = (model.predict(x_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred, digits=4))
