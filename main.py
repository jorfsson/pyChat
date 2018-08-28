# Code is taken from https://medium.com/@david.campion/text-generation-using-bidirectional-lstm-and-doc2vec-models-1-3-8979eb65cb3a
# Solely for experimentation and NN studying

from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Input, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import categorical_accuracy

import spacy
nlp = spacy.load('en')

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

data = (open('./data/sample.txt').read())
chars = sorted(list(set(data)))

id_char = {id:char for id, char in enumerate(chars)}
char_id = {char:id for id, char in enumerate(chars)}

x = []
Y = []
length = len(data)
seq_length = 100

for i in range(0, length - seq_length, 1):
    sequence = data[i:i + seq_length]
    label = data[i + seq_length]
    x.append([char_id[char] for char in sequence])
    Y.append(char_id[label])

x_mod = np.reshape(x, (len(x), seq_length, 1))
x_mod = x_mod / float(len(chars))
y_mod = np_utils.to_categorical(Y)

model = Sequential()
model.add(LSTM(400, input_shape=(x_mod.shape[1], x_mod.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(400))
model.add(Dropout(0.2))
model.add(Dense(y_mod.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(x_mod, y_mod, epochs=1, batch_size=100)
model.save_weights('./models/text_generator_gigantic.h5')

model.load_weights('./models/text_generator_gigantic.h5')

string_mapped = x[99]
# generating characters
for i in range(seq_length):
    x1 = np.reshape(string_mapped,(1,len(string_mapped), 1))
    x1 = x1 / float(len(chars))
    pred_index = np.argmax(model.predict(x1, verbose=0))
    seq = [id_char[value] for value in string_mapped]
    string_mapped.append(pred_index)
    full_string = string_mapped[1:len(string_mapped)]

txt=""
for char in full_string:
    txt = txt+char
print(txt)
