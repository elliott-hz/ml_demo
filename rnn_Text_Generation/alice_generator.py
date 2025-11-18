import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import os

# Check if the text file exists
if not os.path.exists('alice_in_wonderland.txt'):
    print("Error: alice_in_wonderland.txt not found!")
    print("Please make sure the file exists in the current directory.")
    exit(1)

# Open and read the text file
with open('alice_in_wonderland.txt', 'r', encoding='utf-8') as file:
    text = file.read()

print('Text length:', len(text))

# Initialize two lists
segments = []  # used to store character segments of length 60
next_chars = []  # used to store the corresponding next character

# Set the step size and segment length
step = 3
sequence_length = 60

# Iterate over the text to extract segments and their next character
for i in range(0, len(text) - sequence_length, step):
    # Extract a character segment of length 60
    segment = text[i:i + sequence_length]
    # Extract the corresponding next character
    next_char = text[i + sequence_length]
    # Append the segment and the next character to their lists
    segments.append(segment)
    next_chars.append(next_char)

# Print some results for verification
print(f"Number of sequences: {len(segments)}")
print(f"Example segment: {segments[50]}")
print(f"Example next char: {next_chars[50]}")

"""2. Character to Vector

No word embedding is needed here because the model uses character-level tokenization.
Word-level tokenization may have tens of thousands of unique tokens and would require
an embedding to reduce dimensionality, but character-level tokenization has only dozens
or a few hundred unique symbols so one-hot encoding is feasible.
"""

# Build character vocabulary
chars = sorted(list(set(text)))  # get all unique characters
char_to_index = {char: index for index, char in enumerate(chars)}  # map char -> index
index_to_char = {index: char for index, char in enumerate(chars)}  # map index -> char

num_chars = len(char_to_index)
print(f"Number of unique characters: {num_chars}")

num_sequences = len(segments)
# Initialize input tensor and target vectors
X = np.zeros((num_sequences, sequence_length, num_chars), dtype=bool)  # input matrix
y = np.zeros((num_sequences, num_chars), dtype=bool)  # target vector

# Fill input matrix and target vectors
for i, segment in enumerate(segments):
    for t, char in enumerate(segment):
        X[i, t, char_to_index[char]] = 1  # one-hot encoding
    y[i, char_to_index[next_chars[i]]] = 1  # one-hot encoding

# Print sample shapes for verification
print(f"Example input sequence shape: {X[50].shape}")
print(f"Example target character shape: {y[50].shape}")

"""3. Build a neural network"""

model = Sequential([
    Input(shape=(sequence_length, num_chars)),  # input shape: (sequence_length, num_chars)
    LSTM(128),  # LSTM hidden state dimension: 128
    Dense(num_chars, activation="softmax")  # Dense output of size num_chars, softmax gives a probability distribution
])
model.summary()

optimizer = optimizers.RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

"""4. Train the neural network"""

history = model.fit(X, y, batch_size=128, epochs=5)
model.save("alice_generator.keras")

"""Predict the next character"""

def sample_with_temperature(preds, temperature=1.0):
    """
    Sample a character index from the model's probability distribution using a temperature.
    :param preds: predicted probability distribution from the model
    :param temperature: temperature parameter controlling randomness/diversity
    :return: chosen character index
    """
    preds = np.asarray(preds).astype('float64')
    # Step 1: adjust the sharpness of the probability distribution
    preds = preds ** (1 / temperature)
    # Step 2: normalize the probability distribution
    preds = preds / np.sum(preds)

    # Sample a character index from the adjusted distribution
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Define the text generation function
def generate_text(model, seed_text, length=100, temperature=0.5):
    generated = ''
    generated += seed_text
    sentence = seed_text

    for i in range(length):
        x_pred = np.zeros((1, sequence_length, num_chars))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_index[char]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample_with_temperature(preds, temperature)
        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    return generated

# Prepare seed text
seed_text = "Alice was captured by Queen when the rabbit says: `why do cr"
print(f"Seed text length: {len(seed_text)}")

# Test predicting the next character
X_in = np.zeros((1, 60, num_chars), dtype=bool)  # input matrix
for t, char in enumerate(seed_text):
    X_in[0, t, char_to_index[char]] = 1

preds = model.predict(X_in, verbose=0)[0]  # verbose=0 disables logging; [0] extracts the first (only) batch element
print(f"Prediction shape: {preds.shape}")

next_index = sample_with_temperature(preds, temperature=0.5)
next_char = index_to_char[next_index]
print(f"Next character: {next_char}")

# Generate text 1
print("Generated Text (temperature=0.5):")
generated_text = generate_text(model, seed_text, length=400, temperature=0.5)
print(generated_text)

# Generate text 1
print("\nGenerated Text (temperature=0.3):")
generated_text = generate_text(model, seed_text, length=400, temperature=0.3)
print(generated_text)

seed_text2 = "Alice was beginning to get very tired of sitting by her sist"

# Generate text 2
print("\nGenerated Text 2 (temperature=0.3):")
generated_text = generate_text(model, seed_text2, length=400, temperature=0.3)
print(generated_text)

# Generate text 2
print("\nGenerated Text 2 (temperature=0.5):")
generated_text = generate_text(model, seed_text2, length=400, temperature=0.5)
print(generated_text)