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
import random
import sys
import os
import time
import codecs
import collections
from six.moves import cPickle

data_dir = 'data'
save_dir = 'save'
target_file = 'pride'
vocab_file = os.path.join(save_dir, "words_vocab.pkl")
sequences_step = 1
rnn_size = 256
seq_length = 30
learning_rate = 0.001


#Read data

def create_wordlist(doc):
    wl = []
    for word in doc:
        if word.text not in ("\n", "\n\n", '\u2009', 'xa0'):
            wl.append(word.text.lower())
    return wl

wordlist = []

input_file = os.path.join(data_dir, target_file + ".txt")
#read data
with codecs.open(input_file, "r") as f:
    data = f.read()

#create sentences
doc = nlp(data)
wl = create_wordlist(doc)
wordlist = wordlist + wl

#create the dictionary
# count the number of words
word_counts = collections.Counter(wordlist)

# Mapping from index to word : vocabulary
vocabulary_inv = [x[0] for x in word_counts.most_common()]
vocabulary_inv = list(sorted(vocabulary_inv))

# Mapping from word to index
vocab = {x: i for i, x in enumerate(vocabulary_inv)}

words = [x[0] for x in word_counts.most_common()]

#size of vocabulary
vocab_size = len(words)

#save the words and vocabulary
with open(os.path.join(vocab_file), 'wb') as f:
    cPickle.dump((words, vocab, vocabulary_inv), f)

sequences = []
next_words = []

for i in range(0, len(wordlist) - seq_length, sequences_step):
    sequences.append(wordlist[i: i + seq_length])
    next_words.append(wordlist[i + seq_length])

print('nb sequences: ', len(sequences))

X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
y = np.zeros((len(sequences), vocab_size), dtype=np.bool)
for i, sentence in enumerate(sequences):
    for t, word in enumerate(sentence):
        X[i, t, vocab[word]] = 1
    y[i, vocab[next_words[i]]] = 1

def bidrirectional_lstm_model(seq_length, vocab_size):
    print('Build LSTM model.')
    model = Sequential()
    model.add(Bidirectional(LSTM(rnn_size, activation="relu"), input_shape=(seq_length, vocab_size)))
    model.add(Dropout(0.6))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=learning_rate)
    callbacks=[EarlyStopping(patience=2, monitor='val_loss')]
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
    print("model built!")
    return model


md = bidrirectional_lstm_model(seq_length, vocab_size)
md.summary()

batch_size = 32
num_epochs = 50

callbacks=[EarlyStopping(patience=4, monitor='val_loss'),
            ModelCheckpoint(filepath=save_dir + '/' + 'my_model_gen_sentences.{epoch:02d}-{val_loss:.2f}.hdf5',
            monitor='val_loss', verbose=0, mode='auto', period=2)]

history = md.fit(X, y,
                batch_size=batch_size,
                shuffle=True,
                epochs=num_epochs,
                callbacks=callbacks,
                validation_split=0.1)

md.save(save_dir + "/" + 'my_model_generate_sentences.h5')
