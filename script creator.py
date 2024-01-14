import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, PositionalEncoding, MultiHeadAttention, Dropout, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load OpenSubtitles dataset
dataset, info = tfds.load('open_subtitles/it', with_info=True, split='train')

# Extract dialogue lines from the dataset
film_script_data = [example['en'].numpy().decode('utf-8') for example in tfds.as_numpy(dataset.take(5000))]

# Tokenize the script data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(film_script_data)
total_words = len(tokenizer.word_index) + 1

# Create input sequences and labels
input_sequences = []
for script in film_script_data:
    token_list = tokenizer.texts_to_sequences([script])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Define the Transformer model
embedding_dim = 128
num_heads = 4
dff = 512
num_blocks = 4

input_layer = Input(shape=(max_sequence_length-1,))
embedding_layer = Embedding(total_words, embedding_dim)(input_layer)
positional_encoding_layer = PositionalEncoding(max_sequence_length-1, embedding_dim)(embedding_layer)

x = positional_encoding_layer
for _ in range(num_blocks):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim//num_heads)([x, x, x])
    attn_output = Dropout(0.1)(attn_output)
    x = tf.keras.layers.Add()([x, attn_output])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

x = GlobalAveragePooling1D()(x)
output_layer = Dense(total_words, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, verbose=1)

# Function to generate AI-powered dialogue with attention
def generate_dialogue_with_attention(seed_text, next_words, model, max_sequence_length, tokenizer):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Generate dialogue using the trained model with attention
generated_dialogue = generate_dialogue_with_attention("JANE", 10, model, max_sequence_length, tokenizer)
print("Generated Dialogue with Attention:", generated_dialogue)
