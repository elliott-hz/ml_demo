# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.datasets import imdb
# from sklearn.metrics import classification_report
# import matplotlib.pyplot as plt
# import numpy as np
# import os
#
# # ----------------------------------------------------
# # 1. Reproducibility（随机数固定）
# # ----------------------------------------------------
# tf.random.set_seed(42)
# np.random.seed(42)
#
# # ----------------------------------------------------
# # 2. Load & preprocess data
# # ----------------------------------------------------
# vocab_size = 10000
# max_len = 100   # modern: 更长序列通常有更好的表达能力
#
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
#
# # pad to same length
# x_train = pad_sequences(x_train, maxlen=max_len)
# x_test  = pad_sequences(x_test,  maxlen=max_len)
#
# # validation split
# x_train, x_valid = x_train[5000:], x_train[:5000]
# y_train, y_valid = y_train[5000:], y_train[:5000]
#
# print("Train:", x_train.shape)
# print("Valid:", x_valid.shape)
# print("Test:", x_test.shape)
#
# # ----------------------------------------------------
# # 3. Build model with separate embedding layer for pretraining
# # ----------------------------------------------------
# embedding_dim = 64
# rnn_units = 64
#
# # Create inputs
# inputs = layers.Input(shape=(max_len,))
#
# # Create embedding layer
# embedding_layer = layers.Embedding(vocab_size, embedding_dim, name='embedding')
# x = embedding_layer(inputs)
#
# # Add LSTM and output layers
# x = layers.LSTM(
#         rnn_units,
#         dropout=0.2,
#         recurrent_dropout=0.2
#     )(x)
#
# outputs = layers.Dense(1, activation="sigmoid")(x)
#
# # Create full model
# full_model = models.Model(inputs, outputs, name="full_model")
#
# # Create pretraining model (embedding + projection to predict target)
# # For pretraining, we'll create a simple model that uses the embedding layer
# pretrain_outputs = layers.Dense(vocab_size, activation="softmax", name="pretrain_output")(x)
# pretrain_model = models.Model(inputs, pretrain_outputs, name="pretrain_model")
#
# print("Pretraining Model:")
# pretrain_model.summary()
#
# print("\nFull Model:")
# full_model.summary()
#
# # ----------------------------------------------------
# # 4. Pretrain embedding layer
# # ----------------------------------------------------
# pretrain_model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-3),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )
#
# # For pretraining, we need to create targets that match the vocabulary size
# # We'll use a simplified approach where we predict the next word in the sequence
# def create_pretraining_data(sequences, max_len):
#     # Create inputs and targets for pretraining
#     # Predict next word given current word
#     X_pretrain = []
#     y_pretrain = []
#
#     for seq in sequences:
#         for i in range(len(seq)-1):
#             # Create sequences of length 1 for simplicity
#             if i < len(seq)-1:
#                 X_pretrain.append([seq[i]])
#                 y_pretrain.append(seq[i+1])
#
#     # Limit the size for memory constraints
#     limit = min(100000, len(X_pretrain))
#     return pad_sequences(X_pretrain[:limit], maxlen=1), np.array(y_pretrain[:limit])
#
# # Create pretraining data from training set
# print("Creating pretraining data...")
# X_pretrain, y_pretrain = create_pretraining_data(x_train[:1000], max_len)
# print(f"Pretraining data shape: {X_pretrain.shape}, {y_pretrain.shape}")
#
# # Callbacks for pretraining
# if not os.path.exists("result"):
#     os.makedirs("result")
#
# pretrain_callbacks = [
#     EarlyStopping(
#         monitor="loss",
#         patience=3,
#         restore_best_weights=True
#     ),
#     ReduceLROnPlateau(
#         monitor="loss",
#         factor=0.5,
#         patience=2,
#         min_lr=1e-6
#     )
# ]
#
# # Train embedding layer
# print("Pretraining embedding layer...")
# pretrain_history = pretrain_model.fit(
#     X_pretrain, y_pretrain,
#     epochs=5,
#     batch_size=64,
#     callbacks=pretrain_callbacks,
#     verbose=1
# )
#
# # ----------------------------------------------------
# # 5. Freeze embedding layer and train the rest
# # ----------------------------------------------------
# # Freeze embedding layer
# embedding_layer.trainable = False
#
# # Recompile full model
# full_model.compile(
#     optimizer=tf.keras.optimizers.RMSprop(1e-3),
#     loss="binary_crossentropy",
#     metrics=["accuracy"]
# )
#
# # Callbacks for full training
# full_callbacks = [
#     EarlyStopping(
#         monitor="val_loss",
#         patience=6,
#         restore_best_weights=True
#     ),
#     ReduceLROnPlateau(
#         monitor="val_loss",
#         factor=0.5,
#         patience=4,
#         min_lr=1e-6
#     ),
#     ModelCheckpoint(
#         filepath="result/best_lstm_rnn.keras",
#         monitor="val_loss",
#         save_best_only=True
#     )
# ]
#
# # Train the rest of the model
# print("Training full model with frozen embedding layer...")
# full_history = full_model.fit(
#     x_train, y_train,
#     epochs=50,
#     batch_size=64,
#     validation_data=(x_valid, y_valid),
#     callbacks=full_callbacks
# )
#
# # ----------------------------------------------------
# # 6. Unfreeze embedding layer and fine-tune
# # ----------------------------------------------------
# # Unfreeze embedding layer
# embedding_layer.trainable = True
#
# # Recompile with lower learning rate
# full_model.compile(
#     optimizer=tf.keras.optimizers.RMSprop(1e-4),  # Lower LR for fine-tuning
#     loss="binary_crossentropy",
#     metrics=["accuracy"]
# )
#
# # Fine-tune the whole model
# print("Fine-tuning entire model...")
# fine_tune_history = full_model.fit(
#     x_train, y_train,
#     epochs=10,
#     batch_size=64,
#     validation_data=(x_valid, y_valid),
#     callbacks=full_callbacks
# )
#
# # ----------------------------------------------------
# # 7. Plot curves
# # ----------------------------------------------------
# # Combine histories for plotting
# combined_history = {}
# for key in full_history.history.keys():
#     combined_history[key] = full_history.history[key] + fine_tune_history.history[key]
#
# plt.figure(figsize=(8, 5))
# plt.plot(combined_history['loss'])
# plt.plot(combined_history['val_loss'])
# plt.title("Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend(["train", "val"])
# plt.grid()
# plt.savefig("result/loss_lstm.png")
# plt.close()
#
# plt.figure(figsize=(8, 5))
# plt.plot(combined_history['accuracy'])
# plt.plot(combined_history['val_accuracy'])
# plt.title("Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend(["train", "val"])
# plt.grid()
# plt.savefig("result/accuracy_lstm.png")
# plt.close()
#
# # ----------------------------------------------------
# # 8. Evaluation
# # ----------------------------------------------------
# y_pred = (full_model.predict(x_test) > 0.5).astype("int32")
# print(classification_report(y_test, y_pred, digits=4))