# minimal_keras_nlp_transformer_fixed.py
import os

import keras_nlp  # your environment already has keras_nlp / keras_hub style layers
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# ------------------------------
# Config
# ------------------------------
data_path = "deu.txt"  # manythings.org english\tgerman (lowercase expected)
num_samples = 5000
d_model = 64
num_heads = 4
num_layers = 2
intermediate_dim = 128
dropout = 0.1
batch_size = 32
epochs = 3  # for demo; increase for real training

sos_token = "<sos>"
eos_token = "<eos>"

# ------------------------------
# 1) Load data and build tokenizers
# ------------------------------
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Put manythings 'deu.txt' in this path: {data_path}")

eng_texts = []
ger_texts = []
with open(data_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= num_samples:
            break
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        eng = parts[0].lower().strip()
        ger = parts[1].lower().strip()
        ger = f"{sos_token} {ger} {eos_token}"
        eng_texts.append(eng)
        ger_texts.append(ger)

print("Loaded pairs:", len(eng_texts))

# Tokenizers
eng_tokenizer = Tokenizer(filters='', oov_token="<OOV>")
eng_tokenizer.fit_on_texts(eng_texts)
eng_seqs = eng_tokenizer.texts_to_sequences(eng_texts)
max_eng_len = max(len(s) for s in eng_seqs)

ger_tokenizer = Tokenizer(filters='', oov_token="<OOV>")
ger_tokenizer.fit_on_texts(ger_texts)
ger_seqs = ger_tokenizer.texts_to_sequences(ger_texts)
max_ger_len = max(len(s) for s in ger_seqs)

num_eng_tokens = len(eng_tokenizer.word_index) + 1
num_ger_tokens = len(ger_tokenizer.word_index) + 1

print("num_eng_tokens:", num_eng_tokens, "max_eng_len:", max_eng_len)
print("num_ger_tokens:", num_ger_tokens, "max_ger_len:", max_ger_len)

encoder_input_data = pad_sequences(eng_seqs, maxlen=max_eng_len, padding="post")
decoder_input_data = pad_sequences(ger_seqs, maxlen=max_ger_len, padding="post")

# decoder target = decoder_input shifted left
decoder_target = np.zeros_like(decoder_input_data)
decoder_target[:, :-1] = decoder_input_data[:, 1:]
decoder_target_exp = np.expand_dims(decoder_target, -1)  # shape (batch, t, 1)


# ------------------------------
# 2) Model building utilities
# ------------------------------

class LearnedPositionalEmbedding(layers.Layer):
    """
    Learned positional embeddings that handle dynamic sequence length:
    - contains an internal Embedding table of size (max_seq_len, d_model)
    - at call time, builds positions = tf.range(seq_len) and fetches embeddings
    """

    def __init__(self, max_seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        # use mask_zero=False for positional embedding table
        self.pos_emb = layers.Embedding(input_dim=max_seq_len, output_dim=d_model, name="pos_emb_table")

    def call(self, x):
        # x: (batch, seq_len, d_model) or (batch, seq_len)
        seq_len = tf.shape(x)[1]
        pos_indices = tf.range(start=0, limit=seq_len, dtype=tf.int32)
        pos_embeddings = self.pos_emb(pos_indices)  # (seq_len, d_model)
        pos_embeddings = pos_embeddings[tf.newaxis, ...]  # (1, seq_len, d_model)
        # broadcasting add over batch dimension
        return x + pos_embeddings


# ------------------------------
# 3) Build the encoder & decoder (stacked single-layer API)
# ------------------------------
# Inputs
encoder_inputs = keras.Input(shape=(None,), dtype="int32", name="encoder_inputs")
decoder_inputs = keras.Input(shape=(None,), dtype="int32", name="decoder_inputs")

# Token embeddings (mask_zero so compute_mask works)
enc_token_emb_layer = layers.Embedding(input_dim=num_eng_tokens, output_dim=d_model, mask_zero=True,
                                       name="enc_token_emb")
dec_token_emb_layer = layers.Embedding(input_dim=num_ger_tokens, output_dim=d_model, mask_zero=True,
                                       name="dec_token_emb")

# positional embedding helper (max length = max(max_eng_len, max_ger_len))
max_seq_len = max(max_eng_len, max_ger_len)
positional = LearnedPositionalEmbedding(max_seq_len, d_model)

# embedded tokens
enc_tokens = enc_token_emb_layer(encoder_inputs)  # (batch, seq_enc, d_model)
enc_embeddings = positional(enc_tokens)  # dynamic positions added

dec_tokens = dec_token_emb_layer(decoder_inputs)  # (batch, seq_dec, d_model)
dec_embeddings = positional(dec_tokens)  # dynamic positions added

# compute masks from token embeddings
enc_padding_mask = enc_token_emb_layer.compute_mask(encoder_inputs)  # (batch, seq_enc)
dec_padding_mask = dec_token_emb_layer.compute_mask(decoder_inputs)  # (batch, seq_dec)

# Build stacked encoder (call single-layer TransformerEncoder repeatedly)
x = enc_embeddings
for i in range(num_layers):
    x = keras_nlp.layers.TransformerEncoder(
        num_heads=num_heads,
        intermediate_dim=intermediate_dim,
        dropout=dropout,
        name=f"encoder_layer_{i}"
    )(x)  # The layer will use the incoming mask from previous layers if present
encoder_output = x  # (batch, seq_enc, d_model)

# Build stacked decoder: must pass encoder output and masks into each decoder layer
y = dec_embeddings
for i in range(num_layers):
    # Note: keras_nlp.layers.TransformerDecoder in this environment expects inputs:
    #   (target, context, decoder_padding_mask=..., encoder_padding_mask=...)
    y = keras_nlp.layers.TransformerDecoder(
        num_heads=num_heads,
        intermediate_dim=intermediate_dim,
        dropout=dropout,
        name=f"decoder_layer_{i}"
    )(
        y,
        encoder_output,
        decoder_padding_mask=dec_padding_mask,
        encoder_padding_mask=enc_padding_mask,
    )
decoder_output = y  # (batch, seq_dec, d_model)

# Final linear (no softmax) -> use from_logits=True in loss
logits = layers.Dense(num_ger_tokens, activation=None, name="final_dense")(decoder_output)

model = keras.Model([encoder_inputs, decoder_inputs], logits)
model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
model.summary()

# ------------------------------
# 4) Train (quick demo)
# ------------------------------
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_exp,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1
)

# ------------------------------
# 5) Greedy decode (token-by-token) for inference
# ------------------------------
reverse_ger_index = {idx: word for word, idx in ger_tokenizer.word_index.items()}
# Tokenizer maps words->indices starting at 1; add mappings for special ones if necessary
reverse_ger_index[0] = ""  # pad
sos_id = ger_tokenizer.word_index.get(sos_token)
eos_id = ger_tokenizer.word_index.get(eos_token)


def greedy_translate(input_text, max_len=max_ger_len):
    # Encode input
    seq = eng_tokenizer.texts_to_sequences([input_text.lower()])
    seq = pad_sequences(seq, maxlen=max_eng_len, padding="post")

    # Start token
    target_seq = np.array([[sos_id]], dtype=np.int32)

    for _ in range(max_len):
        # model expects (1, t) decoder input; positional embeddings are dynamic so they match t
        preds = model.predict([seq, target_seq], verbose=0)  # shape (1, t, num_ger_tokens)
        next_id = int(np.argmax(preds[0, -1, :]))
        if next_id == eos_id or next_id == 0:
            break
        target_seq = np.concatenate([target_seq, [[next_id]]], axis=1)

    # Convert ids to words (skip special tokens)
    words = []
    for tok in target_seq[0]:
        if tok in (0, sos_id, eos_id) or tok is None:
            continue
        w = reverse_ger_index.get(int(tok), "")
        if w:
            words.append(w)
    return " ".join(words)


# ------------------------------
# 6) Test a few sentences
# ------------------------------
print("\n=== Translation tests ===")
tests = ["hello", "how are you", "good morning", "where is the bank", "i love you"]
for t in tests:
    try:
        print("EN:", t, "-> DE:", greedy_translate(t))
    except Exception as e:
        print("Error translating:", t, e)
