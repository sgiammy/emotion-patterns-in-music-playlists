import pandas as pd
import numpy as np

import os
import math
import logging

def load_dataset_from_path(path):
  """
  Build a dataset in the form of a pandas dataframe where
  the schema is: <LYRIC_PATH, EMOTION>
  """
  
  rows = list()

  # Traverse the dataset directory
  for root, dirs, files in os.walk(path):
    for f in files:
      fields = f.split('_')
      if len(fields) <= 0:
        logging.warning('Could not process file:', f) 
        break
      
      rows.append([os.path.abspath(os.path.join(path, f)), fields[0]])

  return pd.DataFrame(rows, columns=['Lyric_Path', 'Emotion'])

def split_train_validation(df, val_perc=0.1):
  """
  Splits the given dataframe into training and validation set
  """
  # Shuffle dataframe
  tmp_df = df.reindex(np.random.permutation(df.index))

  # Return two portions of the shuffled dataframe
  idx = math.ceil(len(tmp_df.index) * val_perc)
  return (tmp_df[idx:], tmp_df[:idx])
