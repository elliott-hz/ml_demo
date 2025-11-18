# seq2seq_lstm_en_de_fixed.py
import os

import numpy as np
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# ------------------------------
# Step 0: download data from https://www.manythings.org/anki/
# ------------------------------

# ------------------------------
# Configuration parameters
# ------------------------------
# Data path: German dataset from manythings.org (format: English \t German \t ...)
data_path = "deu.txt"
# Number of samples to use, a small subset for demonstration
num_samples = 10000
# LSTM hidden dimension
latent_dim = 32
# Word embedding dimension
embedding_dim = 128
# Batch size
batch_size = 64
# Number of training epochs
epochs = 8

# ------------------------------
# 1. Load and preprocess data
# ------------------------------
# Lists to store English and German sentences
input_texts = []
target_texts = []

# Check if the data file exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}. Download and place 'deu.txt' here.")

# Define start and end tokens
sos_token = "<sos>"  # Start of Sentence token
eos_token = "<eos>"  # End of Sentence token

# Read the data file line by line
with open(data_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        # Only read the specified number of samples
        if i >= num_samples:
            break
        # Split English and German sentences (by tab character)
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        eng, ger = parts[0], parts[1]
        # Convert to lowercase (optional)
        eng = eng.lower()
        ger = ger.lower()

        # Add start and end tokens before and after the German sentence
        target = sos_token + " " + ger + " " + eos_token

        # Add to the corresponding lists
        input_texts.append(eng)
        target_texts.append(target)

print(f"Loaded {len(input_texts)} sentence pairs.")

# ------------------------------
# 2. Text tokenization and sequencing
# ------------------------------
# Create English tokenizer, preserve special symbols (filters='') to avoid losing <sos>/<eos> tokens
eng_tokenizer = Tokenizer(filters='', oov_token="<OOV>")
# Train the tokenizer on English text
eng_tokenizer.fit_on_texts(input_texts)
# Convert English text to sequences (numeric representation)
eng_sequences = eng_tokenizer.texts_to_sequences(input_texts)
# Calculate the maximum length of English sentences
max_eng_len = max(len(s) for s in eng_sequences)

# Create German tokenizer
ger_tokenizer = Tokenizer(filters='', oov_token="<OOV>")
# Train the tokenizer on German text
ger_tokenizer.fit_on_texts(target_texts)
# Convert German text to sequences
ger_sequences = ger_tokenizer.texts_to_sequences(target_texts)
# Calculate the maximum length of German sentences
max_ger_len = max(len(s) for s in ger_sequences)

# Calculate vocabulary size (+1 to include padding symbol 0)
num_eng_tokens = len(eng_tokenizer.word_index) + 1  # +1 for padding (0)
num_ger_tokens = len(ger_tokenizer.word_index) + 1

print("English tokens:", num_eng_tokens, "Max length:", max_eng_len)
print("German tokens:", num_ger_tokens, "Max length:", max_ger_len)

# Pad sequences to make them have the same length
encoder_input_data = pad_sequences(eng_sequences, maxlen=max_eng_len, padding="post")
decoder_input_data = pad_sequences(ger_sequences, maxlen=max_ger_len, padding="post")

print("Encoder input shape:", encoder_input_data.shape)
print("Decoder input shape:", decoder_input_data.shape)

# Decoder target data is the decoder input shifted left by one position (teacher forcing)
decoder_target_data = np.zeros_like(decoder_input_data)
decoder_target_data[:, :-1] = decoder_input_data[:, 1:]
# Last column remains 0 (padding) -- this is acceptable for sparse_categorical_crossentropy

# ------------------------------
# 3. Build Seq2Seq model (training phase)
# ------------------------------
# Encoder part
# Encoder Layer 1 (Input): receives integer sequences of arbitrary length
encoder_inputs = Input(shape=(None,), name="encoder_inputs")
# Encoder Layer 2 (Embedding): converts integer sequences to dense vector representation
enc_emb = Embedding(input_dim=num_eng_tokens, output_dim=embedding_dim, name="encoder_embedding")(encoder_inputs)
# Encoder Layer 3 (LSTM): returns state (hidden state and cell state)
_, state_h, state_c = LSTM(latent_dim, return_state=True, name="encoder_lstm")(enc_emb)
# Final state of encoder will be the initial state of decoder
encoder_states = [state_h, state_c]

# Decoder part (training phase)
# Decoder Layer 1 (Input): decoder input layer
decoder_inputs = Input(shape=(None,), name="decoder_inputs")
# Decoder Layer 2 (Embedding): decoder embedding layer
dec_embedding_layer = Embedding(input_dim=num_ger_tokens, output_dim=embedding_dim, name="decoder_embedding")
dec_emb = dec_embedding_layer(decoder_inputs)

# Decoder Layer 3 (LSTM): returns sequences and state
dec_lstm_layer = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
decoder_outputs, _, _ = dec_lstm_layer(dec_emb, initial_state=encoder_states)

# Decoder Layer 4 (Dense): maps LSTM output to German vocabulary size, uses softmax to output probability distribution
dec_dense_layer = Dense(num_ger_tokens, activation="softmax", name="decoder_dense")
decoder_outputs = dec_dense_layer(decoder_outputs)

# Build complete training model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name="seq2seq_model_train")
# Compile model: use Adam optimizer and sparse categorical crossentropy loss
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# Display model architecture summary
model.summary()

# ------------------------------
# 5. Build inference model
# ------------------------------
# Encoder model for inference phase: input English sentence, output encoded state
encoder_model = Model(encoder_inputs, encoder_states, name="encoder_model_inference")
encoder_model.summary()

# Decoder model for inference phase
# Decoder state inputs: hidden state and cell state from the previous time step
decoder_states_inputs = [Input(shape=(latent_dim,), name="dec_state_input_h"),
                         Input(shape=(latent_dim,), name="dec_state_input_c")]

# Decoder single time step input: process one token at a time
decoder_single_input = Input(shape=(1,), name="decoder_single_input")  # single time step
# Apply embedding layer (reuse weights from training)
dec_single_emb = dec_embedding_layer(decoder_single_input)
# Decoder LSTM layer (reuse weights from training)
dec_outputs_inf, state_h_inf, state_c_inf = dec_lstm_layer(dec_single_emb, initial_state=decoder_states_inputs)
# Apply dense layer (reuse weights from training)
dec_outputs_inf = dec_dense_layer(dec_outputs_inf)  # output shape: (batch, 1, num_ger_tokens)

# Build decoder model for inference phase
decoder_model = Model(
    [decoder_single_input] + decoder_states_inputs,
    [dec_outputs_inf, state_h_inf, state_c_inf],
    name="decoder_model_inference"
)
decoder_model.summary()

# ------------------------------
# 4. Train model
# ------------------------------
# Decoder target data shape: (number of samples, time steps)
# sparse_categorical_crossentropy expects integer labels, but Keras requires 3D shape when input is (samples, time_steps, 1)
decoder_target_data_expanded = np.expand_dims(decoder_target_data, -1)

# Start training the model
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data_expanded,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.15  # use 15% of data as validation set
)

# ------------------------------
# 6. Build reverse index mapping (id -> word)
# ------------------------------
# Create a mapping dictionary from id to word
reverse_ger_index = {idx: word for word, idx in ger_tokenizer.word_index.items()}
# Note: padding symbol 0 maps to empty string
reverse_ger_index[0] = ''


# Helper function to safely get token id
def get_token_id(token):
    return ger_tokenizer.word_index.get(token, None)


# Get the ids of start and end tokens
sos_id = get_token_id(sos_token)
eos_id = get_token_id(eos_token)
# Check if tokens are correctly inserted
if sos_id is None or eos_id is None:
    raise ValueError("Start or end token not found in German tokenizer. Check token insertion.")


# ------------------------------
# 7. Translation function (inference process)
# ------------------------------
def translate_sentence(input_text, max_len=max_ger_len):
    """
    Translate an English sentence to German.

    Parameters:
    input_text: English sentence to translate
    max_len: maximum length of generated German sentence

    Returns:
    Translated German sentence
    """
    # Preprocess input text
    seq = eng_tokenizer.texts_to_sequences([input_text.lower()])
    # Pad sequence to fixed length
    seq = pad_sequences(seq, maxlen=max_eng_len, padding="post")

    # Encode input sentence to get initial states
    states_value = encoder_model.predict(seq, verbose=0)

    # Use start token as the first input to decoder (shape: (1,1))
    target_seq = np.array([[sos_id]])

    # Store decoded words
    decoded_tokens = []
    # Iterate to generate at most max_len words
    for _ in range(max_len):
        # Decoder predicts probability distribution for next word
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        # Select the word index with highest probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        # Stop if padding symbol or end token is encountered
        if sampled_token_index == 0 or sampled_token_index == eos_id:
            break

        # Get the word corresponding to the token index
        sampled_word = reverse_ger_index.get(sampled_token_index, "")
        if sampled_word == "":
            break

        # Add to decoding results
        decoded_tokens.append(sampled_word)

        # Update decoder input (previously sampled word) and states
        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]

    # Join word list into sentence
    decoded_sentence = " ".join(decoded_tokens)
    return decoded_sentence.strip()


# ------------------------------
# 8. Test translation functionality
# ------------------------------
print("\n=== Translation Test ===")
# Test cases
examples = [
    "how are you?",
    "i love you",
    "what is your name?",
    "where is the bank?"
]

# Translate each example and print results
for s in examples:
    print("English:", s)
    print("German (pred):", translate_sentence(s))
    print("---")
