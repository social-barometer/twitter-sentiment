import pandas as pd
import sklearn 

def clean(path='./training_data/Jan9-2012-tweets-clean.txt'):
  """Cleans the data. Because she hasn't had a bath in a year.
  
  Arguments:
    path (str): Path to the data file

  Returns:
    A pandas data frame with clean data.
  """
  df = pd.read_csv(path,
                   sep=':',
                   header=None,
                   error_bad_lines=False)
  df = df.drop([0, 2, 4], 1)
  df = df.dropna()
  df = df.rename(columns={1: 'tweet', 3: 'emotion'})
  return df

