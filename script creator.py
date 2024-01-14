import requests
from bs4 import BeautifulSoup
import re
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Function to fetch film script data from a URL
def fetch_script_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')  # Adjust based on the HTML structure of the script
    script_text = ' '.join([p.get_text() for p in paragraphs])
    return script_text
script_urls = [
    'https://imsdb.com/',
    'https://moviescriptsandscreenplays.com/',
]

film_script_data = [fetch_script_data(url) for url in script_urls]

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

# attention mechanism
embedding_dim = 50
units = 128

input_layer = Input(shape=(max_sequence_length-1,))
embedding_layer = Embedding(total_words, embedding_dim, input_length=max_sequence_length-1)(input_layer)
lstm_layer = LSTM(units, return_sequences=True)(embedding_layer)
attention = Attention()([lstm_layer, lstm_layer])
context_vector = tf.reduce_sum(attention * lstm_layer, axis=1)
context_vector = tf.expand_dims(context_vector, axis=1)
merged = tf.keras.layers.Concatenate(axis=-1)([lstm_layer, context_vector])
lstm_output = LSTM(units)(merged)
output_layer = Dense(total_words, activation='softmax')(lstm_output)
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=50, verbose=1)

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

