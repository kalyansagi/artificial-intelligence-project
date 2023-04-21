#Importing the libraries
import tensorflow as tf
import string
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.applications.densenet import layers
from keras.utils import pad_sequences

data = open('./resources/dataset1.txt', encoding="utf8").read().splitlines()

token = Tokenizer()
token.fit_on_texts(data)

encoded_text = token.texts_to_sequences(data)
# vocabulary size should be + 1
vocab_size = len(token.word_counts) + 1


datalist = []
for d in encoded_text:
  if len(d)>1:
    for i in range(2, len(d)):
      datalist.append(d[:i])
      print(d[:i])

max_length = 20
sequences = pad_sequences(datalist, maxlen=max_length, padding='pre')
X = sequences[:, :-1]
y = sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))

#To run with SimpleRNN
# model.add(layers.GRU(100, return_sequences=True))
# model.add(layers.SimpleRNN(100))

#To run with LSTM
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))

model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, batch_size=32, epochs=100)

#Evaluation the model
print("Model Accuracy: "+str(history.history['accuracy'][len(history.history['accuracy'])-1]))
print("Model Loss: "+str(history.history['loss'][len(history.history['loss'])-1]))


#Visualizing the performance metrics
acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.title('Training accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.title('Training Loss')
plt.legend()

plt.show()

def generate_poetry(seed_text, n_lines):
  for i in range(n_lines):
    text = []
    for _ in range(poetry_length):
      encoded = token.texts_to_sequences([seed_text])
      encoded = pad_sequences(encoded, maxlen=seq_length, padding='pre')

      y_pred = np.argmax(model.predict(encoded, verbose=0), axis=-1)

      predicted_word = ""
      for word, index in token.word_index.items():
        if index == y_pred:
          predicted_word = word
          break

      seed_text = seed_text + ' ' + predicted_word
      text.append(predicted_word)

    seed_text = text[-1]
    text = ' '.join(text)
    print(text)

poetry_length = 10
seed_text = 'i love you'
generate_poetry(seed_text, 5)