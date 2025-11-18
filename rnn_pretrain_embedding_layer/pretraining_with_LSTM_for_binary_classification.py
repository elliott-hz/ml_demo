import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------------------------------
# 1. Reproducibility（随机数固定）
# ----------------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)

# ----------------------------------------------------
# 2. Load & preprocess data
# ----------------------------------------------------
vocab_size = 10000
max_len = 50  # modern: 更长序列通常有更好的表达能力

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# pad to same length
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# validation split
x_train, x_valid = x_train[5000:], x_train[:5000]
y_train, y_valid = y_train[5000:], y_train[:5000]

print("Train:", x_train.shape)
print("Valid:", x_valid.shape)
print("Test:", x_test.shape)

# ====================================================
# STAGE 1: PRE-TRAIN EMBEDDING LAYER SEPARATELY
# ====================================================
print("\n" + "=" * 60)
print("STAGE 1: PRE-TRAINING EMBEDDING LAYER")
print("=" * 60)

embedding_dim = 64
rnn_units = 32

# Build a simple model for embedding pre-training
# Input -> Embedding -> GlobalAveragePooling -> Dense -> Output
embedding_input = layers.Input(shape=(max_len,))
embedding_output = layers.Embedding(vocab_size, embedding_dim, name="embedding_layer")(embedding_input)
pooled = layers.GlobalAveragePooling1D()(embedding_output)
pretrain_output = layers.Dense(1, activation="sigmoid")(pooled)

pretrain_model = models.Model(embedding_input, pretrain_output)
pretrain_model.summary()

# Compile pre-training model
pretrain_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Pre-training callbacks
pretrain_callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        filepath="result/pretrain_embedding.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

if not os.path.exists("result"):
    os.makedirs("result")

# Train embedding layer
print("\nTraining Embedding layer...")
pretrain_history = pretrain_model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_valid, y_valid),
    callbacks=pretrain_callbacks,
    verbose=1
)

# Extract the trained embedding layer
trained_embedding_layer = pretrain_model.get_layer("embedding_layer")
print("\n✓ Embedding layer pre-trained successfully!")

# ====================================================
# STAGE 2: BUILD FINAL MODEL WITH FROZEN EMBEDDING
# ====================================================
print("\n" + "=" * 60)
print("STAGE 2: BUILDING FINAL MODEL WITH FROZEN EMBEDDING")
print("=" * 60)

# Build final model
inputs = layers.Input(shape=(max_len,))

# Use the pre-trained embedding layer and freeze it
trained_embedding_layer.trainable = False # Freeze embedding layer
x = trained_embedding_layer(inputs)

x = layers.LSTM(
    rnn_units,
    dropout=0.2,
    recurrent_dropout=0.2
)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)
model.summary()

# Compile final model
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ====================================================
# STAGE 3: TRAIN FINAL MODEL (LSTM + DENSE)
# ====================================================
print("\n" + "=" * 60)
print("STAGE 3: TRAINING LSTM + DENSE (EMBEDDING FROZEN)")
print("=" * 60)

# Callbacks for final training
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

# Train final model with frozen embedding
print("\nTraining final model (LSTM + Dense with frozen Embedding)...")
history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=64,
    validation_data=(x_valid, y_valid),
    callbacks=callbacks,
    verbose=1
)

# ====================================================
# PLOT PRE-TRAINING RESULTS
# ====================================================
print("\n" + "=" * 60)
print("PLOTTING PRE-TRAINING RESULTS")
print("=" * 60)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(pretrain_history.history['loss'], label='train loss')
plt.plot(pretrain_history.history['val_loss'], label='val loss')
plt.title("Embedding Pre-Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(pretrain_history.history['accuracy'], label='train accuracy')
plt.plot(pretrain_history.history['val_accuracy'], label='val accuracy')
plt.title("Embedding Pre-Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("result/pretrain_curves.png")
plt.close()

# ====================================================
# PLOT FINAL TRAINING RESULTS
# ====================================================
print("\nPlotting final training results...")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title("Final Model Training Loss (Frozen Embedding)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title("Final Model Training Accuracy (Frozen Embedding)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("result/final_training_curves.png")
plt.close()

# ====================================================
# STAGE 4: EVALUATION AND PREDICTION
# ====================================================
print("\n" + "=" * 60)
print("STAGE 4: EVALUATION AND PREDICTION")
print("=" * 60)

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Make predictions
print("\nMaking predictions on test set...")
y_pred_probs = model.predict(x_test, verbose=0)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Show sample predictions
print("\nSample Predictions (first 20 samples):")
print("-" * 60)
for i in range(min(20, len(y_test))):
    true_label = "Positive" if y_test[i] == 1 else "Negative"
    pred_label = "Positive" if y_pred[i] == 1 else "Negative"
    confidence = y_pred_probs[i][0]
    status = "✓" if y_test[i] == y_pred[i] else "✗"
    print(f"{status} Sample {i:2d}: True={true_label:8s}, Pred={pred_label:8s}, Confidence={confidence:.4f}")

print("\n" + "=" * 60)
print("Training completed successfully!")
print("=" * 60)
