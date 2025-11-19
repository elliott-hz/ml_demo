# Simple Transformer for English to German Translation
# Simplified version for learning purposes

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# ------------------------------
# Configuration parameters
# ------------------------------
data_path = "deu.txt"  # German dataset from manythings.org
num_samples = 10000  # Small subset for faster training
d_model = 64  # Smaller model dimension
num_heads = 4  # Fewer attention heads
num_layers = 2  # Only 2 encoder/decoder layers
dropout_rate = 0.1
batch_size = 32
epochs = 3  # Fewer epochs for quick training

# ------------------------------
# 1. Load and preprocess data
# ------------------------------
input_texts = []
target_texts = []

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}. Download and place 'deu.txt' here.")

# Define start and end tokens
sos_token = "<sos>"
eos_token = "<eos>"

with open(data_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= num_samples:
            break
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        eng, ger = parts[0], parts[1]
        eng = eng.lower()
        ger = ger.lower()

        # Add start and end tokens to German sentences
        target = sos_token + " " + ger + " " + eos_token

        input_texts.append(eng)
        target_texts.append(target)

print(f"Loaded {len(input_texts)} sentence pairs.")

# ------------------------------
# 2. Text tokenization and sequencing
# ------------------------------
# Create tokenizers
eng_tokenizer = Tokenizer(filters='', oov_token="<OOV>")
eng_tokenizer.fit_on_texts(input_texts)
eng_sequences = eng_tokenizer.texts_to_sequences(input_texts)
max_eng_len = max(len(s) for s in eng_sequences)

ger_tokenizer = Tokenizer(filters='', oov_token="<OOV>")
ger_tokenizer.fit_on_texts(target_texts)
ger_sequences = ger_tokenizer.texts_to_sequences(target_texts)
max_ger_len = max(len(s) for s in ger_sequences)

# Vocabulary sizes
num_eng_tokens = len(eng_tokenizer.word_index) + 1
num_ger_tokens = len(ger_tokenizer.word_index) + 1

print("English tokens:", num_eng_tokens, "Max length:", max_eng_len)
print("German tokens:", num_ger_tokens, "Max length:", max_ger_len)

# Pad sequences
encoder_input_data = pad_sequences(eng_sequences, maxlen=max_eng_len, padding="post")
decoder_input_data = pad_sequences(ger_sequences, maxlen=max_ger_len, padding="post")

print("Encoder input shape:", encoder_input_data.shape)
print("Decoder input shape:", decoder_input_data.shape)

# Decoder target data (shifted left)
decoder_target_data = np.zeros_like(decoder_input_data)
decoder_target_data[:, :-1] = decoder_input_data[:, 1:]
print("Decoder target shape:", decoder_target_data.shape)


# ------------------------------
# 3. Simple Transformer Building Blocks
# ------------------------------

def positional_encoding(position, d_model):
    """Simple positional encoding using sine and cosine functions"""
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (
            2 * (np.arange(d_model)[np.newaxis, :]) // 2) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


class SimpleEncoderLayer(tf.keras.layers.Layer):
    """Simplified Transformer Encoder Layer"""

    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super(SimpleEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.dropout = Dropout(dropout_rate)
        self.layernorm = LayerNormalization(epsilon=1e-6)

    def call(self, x, training, mask=None):
        # Multi-head attention
        attn_output = self.mha(x, x, x, attention_mask=mask)
        attn_output = self.dropout(attn_output, training=training)
        # Add & Norm
        out = self.layernorm(x + attn_output)
        return out


class SimpleDecoderLayer(tf.keras.layers.Layer):
    """Simplified Transformer Decoder Layer"""

    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super(SimpleDecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)  # Self attention
        self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)  # Encoder-decoder attention
        self.dropout = Dropout(dropout_rate)
        self.layernorm = LayerNormalization(epsilon=1e-6)

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        # Self attention with look ahead mask
        attn1 = self.mha1(x, x, x, attention_mask=look_ahead_mask)
        attn1 = self.dropout(attn1, training=training)
        out1 = self.layernorm(x + attn1)

        # Encoder-decoder attention
        attn2 = self.mha2(out1, enc_output, enc_output, attention_mask=padding_mask)
        attn2 = self.dropout(attn2, training=training)
        out2 = self.layernorm(out1 + attn2)

        return out2


def create_padding_mask(seq):
    """Create padding mask for attention"""
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """Create look ahead mask for decoder self-attention"""
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


# ------------------------------
# 4. Build Simple Transformer Model
# ------------------------------
# Encoder inputs
encoder_inputs = Input(shape=(None,), name="encoder_inputs")
enc_padding_mask = tf.keras.layers.Lambda(create_padding_mask)(encoder_inputs)

# Encoder embedding + positional encoding
enc_embedding = Embedding(input_dim=num_eng_tokens, output_dim=d_model)(encoder_inputs)
enc_embedding *= tf.math.sqrt(tf.cast(d_model, tf.float32))
# Create positional encoding with a size that can accommodate both encoder and decoder
max_seq_len = max(max_eng_len, max_ger_len)
pos_encoding = positional_encoding(max_seq_len, d_model)
encoder_x = enc_embedding + pos_encoding[:, :max_eng_len, :]
encoder_x = Dropout(dropout_rate)(encoder_x)

# Encoder layers (simplified)
encoder_layers = []
for _ in range(num_layers):
    encoder_layer = SimpleEncoderLayer(d_model, num_heads, dropout_rate)
    encoder_x = encoder_layer(encoder_x, training=True, mask=enc_padding_mask)
    encoder_layers.append(encoder_layer)

# Decoder inputs
decoder_inputs = Input(shape=(None,), name="decoder_inputs")
dec_padding_mask = tf.keras.layers.Lambda(create_padding_mask)(decoder_inputs)
look_ahead_mask = tf.keras.layers.Lambda(lambda x: create_look_ahead_mask(tf.shape(x)[1]))(decoder_inputs)
combined_mask = tf.keras.layers.Lambda(lambda x: tf.maximum(x[0], x[1]))([dec_padding_mask, look_ahead_mask])

# Decoder embedding + positional encoding
dec_embedding = Embedding(input_dim=num_ger_tokens, output_dim=d_model)(decoder_inputs)
dec_embedding *= tf.math.sqrt(tf.cast(d_model, tf.float32))
decoder_x = dec_embedding + pos_encoding[:, :max_ger_len, :]
decoder_x = Dropout(dropout_rate)(decoder_x)

# Decoder layers (simplified)
decoder_layers = []
for _ in range(num_layers):
    decoder_layer = SimpleDecoderLayer(d_model, num_heads, dropout_rate)
    decoder_x = decoder_layer(
        decoder_x, encoder_x, training=True, look_ahead_mask=combined_mask, padding_mask=enc_padding_mask)
    decoder_layers.append(decoder_layer)

# Final output layer
decoder_output = Dense(num_ger_tokens, activation="softmax")(decoder_x)

# Build model
model = Model([encoder_inputs, decoder_inputs], decoder_output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ------------------------------
# 5. Train model
# ------------------------------
decoder_target_data_expanded = np.expand_dims(decoder_target_data, -1)

model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data_expanded,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2
)

# ------------------------------
# 6. Translation (prediction) function
# ------------------------------

# Build reverse index mapping (id -> word)
reverse_ger_index = {idx: word for word, idx in ger_tokenizer.word_index.items()}
reverse_ger_index[0] = ''  # Padding symbol maps to empty string


def get_token_id(token, tokenizer):
    """Helper function to safely get token id"""
    return tokenizer.word_index.get(token, None)


# Get the ids of start and end tokens
sos_id = get_token_id(sos_token, ger_tokenizer)
eos_id = get_token_id(eos_token, ger_tokenizer)


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
    seq = pad_sequences(seq, maxlen=max_eng_len, padding="post")

    # Encoder processing (same as during training)
    enc_padding_mask = create_padding_mask(seq)
    enc_embedding_layer = model.get_layer("embedding")
    enc_embedding = enc_embedding_layer(seq)
    enc_embedding *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    encoder_x = enc_embedding + pos_encoding[:, :max_eng_len, :]
    encoder_x = Dropout(dropout_rate)(encoder_x, training=False)

    # Apply encoder layers
    for encoder_layer in encoder_layers:
        encoder_x = encoder_layer(encoder_x, training=False, mask=enc_padding_mask)

    encoder_output = encoder_x

    # Start with SOS token
    target_seq = np.array([[sos_id]])

    # Generate translation token by token
    for _ in range(max_len):
        # Create masks for decoder
        dec_padding_mask = create_padding_mask(target_seq)
        look_ahead_mask = create_look_ahead_mask(tf.shape(target_seq)[1])
        combined_mask = tf.maximum(dec_padding_mask, look_ahead_mask)

        # Decoder processing
        dec_embedding_layer = model.get_layer("embedding_1")
        dec_embedding = dec_embedding_layer(target_seq)
        dec_embedding *= tf.math.sqrt(tf.cast(d_model, tf.float32))
        decoder_x = dec_embedding + pos_encoding[:, :tf.shape(target_seq)[1], :]
        decoder_x = Dropout(dropout_rate)(decoder_x, training=False)

        # Apply decoder layers
        decoder_output_temp = decoder_x
        for decoder_layer in decoder_layers:
            decoder_output_temp = decoder_layer(
                decoder_output_temp, encoder_output, training=False,
                look_ahead_mask=combined_mask, padding_mask=enc_padding_mask)

        # Final output layer
        final_output_layer = model.get_layer("dense")
        decoder_output_result = final_output_layer(decoder_output_temp)

        # Get the most likely next token
        predicted_id = np.argmax(decoder_output_result[0, -1, :])

        # Stop if we've reached the end token or padding
        if predicted_id == eos_id or predicted_id == 0:
            break

        # Add predicted token to sequence
        target_seq = np.concatenate([target_seq, [[predicted_id]]], axis=1)

    # Convert token IDs back to words
    translated_words = []
    for token_id in target_seq[0]:
        if token_id not in [0, sos_id, eos_id]:  # Skip special tokens
            word = reverse_ger_index.get(token_id, '')
            if word:
                translated_words.append(word)

    return ' '.join(translated_words)


# ------------------------------
# 7. Test translation functionality
# ------------------------------
print("\n=== Translation Test ===")
test_sentences = [
    "hello",
    "how are you",
    "i love you",
    "good morning",
    "where is the bank"
]

print("Testing translation functionality:")
for sentence in test_sentences:
    try:
        translation = translate_sentence(sentence)
        print(f"English: {sentence}")
        print(f"German:  {translation}")
        print("-" * 30)
    except Exception as e:
        print(f"Error translating '{sentence}': {e}")
        print("-" * 30)

print("\nTraining completed! This simplified version focuses on the core Transformer architecture:")
print("- 2 encoder layers and 2 decoder layers (instead of 6)")
print("- Smaller model dimensions (64 instead of 512)")
print("- Fewer attention heads (4 instead of 8)")
print("- Limited dataset (1000 samples instead of 10000)")
print("- Fewer training epochs (10 instead of 20 or more)")
