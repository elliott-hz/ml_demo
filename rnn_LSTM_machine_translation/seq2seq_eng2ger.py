import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# 1. Load & preprocess data
# ------------------------------
# Download dataset:
# https://www.manythings.org/anki/deu-eng.zip
# Extract file: deu.txt  (tab separated)
data_path = "deu.txt"

input_texts = []
target_texts = []

num_samples = 10000  # small subset, enough for demo

with open(data_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= num_samples:
            break
        eng, ger, _ = line.split("\t")

        # add start and end tokens for decoder
        target = "\t" + ger + "\n"

        input_texts.append(eng.lower())
        target_texts.append(target.lower())

# ------------------------------
# 2. Tokenization
# ------------------------------
eng_tokenizer = Tokenizer(filters='')
eng_tokenizer.fit_on_texts(input_texts)
eng_sequences = eng_tokenizer.texts_to_sequences(input_texts)
max_eng_len = max(len(s) for s in eng_sequences)

ger_tokenizer = Tokenizer(filters='')
ger_tokenizer.fit_on_texts(target_texts)
ger_sequences = ger_tokenizer.texts_to_sequences(target_texts)
max_ger_len = max(len(s) for s in ger_sequences)

num_eng_tokens = len(eng_tokenizer.word_index) + 1
num_ger_tokens = len(ger_tokenizer.word_index) + 1

encoder_input_data = pad_sequences(eng_sequences, maxlen=max_eng_len, padding="post")
decoder_input_data = pad_sequences(ger_sequences, maxlen=max_ger_len, padding="post")

# decoder target shifts by one
decoder_target_data = np.zeros_like(decoder_input_data)
decoder_target_data[:, :-1] = decoder_input_data[:, 1:]

# ------------------------------
# 3. Build Seq2Seq model
# ------------------------------
latent_dim = 256

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(num_eng_tokens, 128)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb = Embedding(num_ger_tokens, 128)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(num_ger_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.summary()

# ------------------------------
# 4. Train
# ------------------------------
model.fit(
    [encoder_input_data, decoder_input_data],
    np.expand_dims(decoder_target_data, -1),
    batch_size=64,
    epochs=20,
    validation_split=0.1
)

# ------------------------------
# 5. Build inference model
# ------------------------------
# Encoder model
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder model
dec_state_input_h = Input(shape=(latent_dim,))
dec_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [dec_state_input_h, dec_state_input_c]

dec_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(
    dec_emb, initial_state=decoder_states_inputs
)
dec_states_inf = [state_h_inf, state_c_inf]
dec_outputs_inf = decoder_dense(dec_outputs_inf)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [dec_outputs_inf] + dec_states_inf
)

# word index reverse mapping
reverse_ger_index = {i: w for w, i in ger_tokenizer.word_index.items()}

# ------------------------------
# 6. Inference function
# ------------------------------
def translate_sentence(input_text):
    seq = eng_tokenizer.texts_to_sequences([input_text.lower()])
    seq = pad_sequences(seq, maxlen=max_eng_len, padding="post")

    states = encoder_model.predict(seq, verbose=0)
    target_seq = np.array([[ger_tokenizer.word_index["\t"]]])

    decoded_sentence = ""

    for _ in range(max_ger_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states, verbose=0)

        sampled_token = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_ger_index.get(sampled_token, '')

        if sampled_word == "\n":
            break

        decoded_sentence += sampled_word + " "

        target_seq = np.array([[sampled_token]])
        states = [h, c]

    return decoded_sentence.strip()


# ------------------------------
# 7. Test translation
# ------------------------------
print("\n=== Translation Test ===")
test_sentence = "how are you?"
print("English:", test_sentence)
print("German (pred):", translate_sentence(test_sentence))
