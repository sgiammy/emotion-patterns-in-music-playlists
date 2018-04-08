import numpy as np
import pandas as pd

import spacy
from spacy.language import Language

import sys
import glob
import os
import pickle

from collections import Counter

import logging

from sklearn.svm import SVC

class LyricsSVM:
  def __init__(self):
    # Basic init stuff
    self.labels = ['happy', 'sad', 'relaxed', 'angry']
    self.target_model = list()

  def build_lang(self, vec_path):
    """
    Build (or read if it was already built) the language vector model
    used to classify our lyrics
    """
    
    # Let's use some pretrained simple model
    self.nlp = spacy.load('en_core_web_md')
    logging.info('Language model successfully loaded')
    """
    self.nlp = Language()
    with open(vec_path, 'rb') as file_:
       header = file_.readline()
       nr_row, nr_dim = header.split()
       self.nlp.vocab.reset_vectors(width=int(nr_dim))
       for line in file_:
           line = line.rstrip().decode('utf8')
           pieces = line.rsplit(' ', int(nr_dim))
           word = pieces[0]
           vector = np.asarray([float(v) for v in pieces[1:]], dtype='f')
           self.nlp.vocab.set_vector(word, vector)  # add the vectors to the vocab
    logging.info('Language model successfully loaded')
    """

  def train(self, df, **params):
    """
    Same as the below build model function but uses a dataframe instead
    of paths to lyrics
    """
    self.target_model = list()
    self.labels = df.Emotion.unique()

    # Turn emotion labels into numerical features
    mapping = dict(zip(self.labels, range(len(self.labels)))) # "happy": 0, "sad": 1, ...
    df['Emotion'].map(mapping)

    # Read files corresponding to each available emotion
    rows = list()
    for index, row in df.iterrows():
      lyric = row['Lyric_Path']
      emotion = row['Emotion']
      with open(lyric, 'r') as lyric_file: 
        doc = self.nlp(lyric_file.read())
        doc = self._preprocess(doc)
        if len(doc.vector) != 300: # TODO: this is not good of course
          rows.append((doc.vector, emotion))
    
    # Build dataset dataframe with schema <Doc_Vector, Emotion>
    self.lyrics_dataset = pd.DataFrame(rows, columns=['Vector', 'Emotion'])

    # Extract data
    X = self.lyrics_dataset['Vector'].as_matrix().reshape(-1, 1)
    y = self.lyrics_dataset['Emotion'].as_matrix().reshape(-1, 1)

    print(X.shape)

    # Create and train our model
    self.target_model = SVC(params)
    self.target_model.fit(X, y)

    logging.info('Model successfully trained')

  def _preprocess(self, doc):
    """
    Perform some documents preprocessing
    """
    # Just remove stopwords
    tks = list(filter(lambda tk: not tk.is_stop, doc))
    return spacy.tokens.Doc(self.nlp.vocab, words=[tk.text for tk in tks])

  ## TODO, need to update this one
  def build_model(self, training_path):
    """
    Build our model which is a dictionary where the key is the target label
    and the value is the vector value associated to that target
    """
    self.target_dict = dict()

    # Read files corresponding to each available emotion
    for emotion in self.labels:
      path = os.path.join(training_path, '{}_*'.format(emotion))
      # Compute sum of each lyrics document vector norm
      count, value = 0, 0
      for lyric in glob.glob(path):
        with open(lyric, 'r') as lyric_file:
          doc = self.nlp(lyric_file.read()) 
          value += doc.vector_norm
          count += 1
      # Add field to the target dictionary
      self.target_dict[emotion] = value/count

    logging.info('Model successfully built from filesystem: {}'.format(self.target_dict))

  def predict(self, path_to_lyric, k=5):
    """
    Predict the emotion for the given lyric.
    @path_to_lyric should be a valid path to a lyric
    """
    with open(path_to_lyric, 'r') as l:
      # Build doc object
      doc = self.nlp(l.read())
      # Obtain vector's norm
      value = doc.vector
      # Return the predicted class
      label = self.labels[self.target_model(value)]
      logging.info('Prediction for {}: {}'.format(path_to_lyric, label))
      return label

  def persist_target_model(self, target_path):
    """
    Persiste the built target model in a file
    """
    if self.target_model is None:
      logging.warning('Target model does not exist yet')
      return
    if os.path.exists(target_path):
      logging.warning('Model file already exists')
      return
    # Create the model's file
    with open(target_path, 'wb') as model_file:
      pickle.dump(self.target_model, model_file)
      logging.info('Model successfully persisted ({})'.format(target_path))

  def load_model(self, target_path):
    """
    Load serialized model object
    """
    with open(target_path, 'rb') as model_file:
      self.target_model = pickle.load(model_file)

  def score(self, testDf):
    """
    Used to check model's performances. Return an accuracy value
    """
    count, correct = 0, 0
    for index, row in testDf.iterrows():
      lyric = row['Lyric_Path']
      emotion = row['Emotion']
      label = self.predict(lyric)
      if label == emotion:
        correct += 1
      count += 1
    return correct / count
