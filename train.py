import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Activation, LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from data.preprocess import clean, tokenize, remove_common_words, undersample

EMBEDDINGS_DIM = 25
MAX_SEQUENCE_LENGTH = 280 # Max tweet length
MAX_NB_WORDS = 30000

df = clean('./Jan9-2012-tweets-clean.txt')
df = undersample(df, 'joy', 3000)

X = df['tweet']
y = df['emotion']

word_index, sequences = tokenize(X, MAX_NB_WORDS)
word_index, sequences = remove_common_words(word_index, sequences, 100)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

label_encoder = LabelEncoder()
label_encoder.fit(y)

print('Labels:', label_encoder.classes_)
print('Labels encodings:', label_encoder.transform(label_encoder.classes_))

labels = to_categorical(label_encoder.transform(y))

print("Shape of data: {}.".format(data.shape))

embeddings_index = {}
with open('./glove.twitter.27B/glove.twitter.27B.{}d.txt'.format(EMBEDDINGS_DIM)) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found {} word vectors.'.format(len(embeddings_index)))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDINGS_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


model = Sequential()
model.add(Embedding(
    len(word_index) + 1,
    EMBEDDINGS_DIM,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=False
))
model.add(Dropout(0.4))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128, return_sequences=False))
model.add(Activation('relu'))
model.add(Dense(6))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

filepath="./models/lstm-{epoch:02d}-{loss:0.3f}-{acc:0.3f}-{val_loss:0.3f}-{val_acc:0.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001)

print(model.summary())
model.fit(
    data,
    labels,
    batch_size=128,
    epochs=3,
    validation_split=0.2,
    shuffle=True,
    callbacks=[checkpoint, reduce_lr]
)
