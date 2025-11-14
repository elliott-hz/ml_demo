from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

texts_train = [
    "This is a good movie", "I really love this funny movie", "This movie is not good"
]

# 1. Tokenizer
vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(texts_train)

# 2. Text to Sequence
sequences_train = tokenizer.texts_to_sequences(texts_train)

print("word_index:", tokenizer.word_index)
print("\nsequences:", sequences_train)

# 3. Padding
maxlen = 20
x_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)

print("\nx_train shape:", x_train.shape)
print(x_train)
