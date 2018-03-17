import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

def clean(path):
  """Cleans the data. Because she hasn't had a bath in a year.

  Arguments:
    path (str): Path to the data file

  Returns:
    A pandas data frame with clean data.
  """
  df = pd.read_csv(path,
                   sep=':',
                   header=None,
                   error_bad_lines=False,
                   warn_bad_lines=False)
  df = df.drop([0, 2, 4], 1)
  df = df.dropna()
  df = df.rename(columns={1: 'tweet', 3: 'emotion'})
  df['emotion'] = df['emotion'].str.strip()
  df['tweet'] = df['tweet'].str.strip()

  return df

def tokenize(texts, num_words=None):
  """Tokenizes given texts and returns the word index and sequences.

  Arguments:
    texts: pandas series of texts
    num_words (int): How many words to include in the index
  Returns:
    Word index and sequences
  """
  tokenizer = Tokenizer(num_words=num_words)
  tokenizer.fit_on_texts(texts)
  sequences = tokenizer.texts_to_sequences(texts)

  return tokenizer.word_index, sequences

def remove_common_words(word_index, sequences, n):
    """Removes the n most common words from the word_index."""

    # new_index {word: index - n for word, index in word_index.items()
    #            if index > n}

    common_word_indexes = []
    new_index = {}

    for word, index in word_index.items():
      if index > n:
        new_index[word] = index - n
      else:
        common_word_indexes.append(index)

    new_seqs = []
    for seq in sequences:
      new_seq = []
      for index in seq:
        if index not in common_word_indexes:
          new_seq.append(index - n)
      new_seqs.append(new_seq)


    return new_index, new_seqs

def undersample(df, label, sample_size):
    """Undersample the all the rows with given label in the df."""

    df_label = df[df['emotion'] == label]
    df_no_label = df[df['emotion'] != label]

    df_label_undersampled = resample(
        df_label,
        replace=True,
        n_samples=sample_size,
        random_state=313
    )

    undersampled = pd.concat([df_no_label, df_label_undersampled])
    return undersampled.sample(frac=1) # Shuffle
