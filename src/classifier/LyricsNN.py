import numpy

import spacy
from spacy.language import Language

import sys
import glob
import os
import pickle

import logging

class LyricsNN:
  def __init__(self):
    # Basic init stuff
    self.labels = ['happy', 'sad', 'relaxed', 'angry']
    self.target_dict = None

  def build_lang(self, vec_path):
    """
    Build (or read if it was already built) the language vector model
    used to classify our lyrics
    """
    
    # Let's use some pretrained simple model
    self.nlp = spacy.load('en_core_web_md')
    logging.info('Language model successfully loaded')
    """
    with open(vec_path, 'rb') as file_:
       header = file_.readline()
       nr_row, nr_dim = header.split()
       nlp.vocab.reset_vectors(width=int(nr_dim))
       for line in file_:
           line = line.rstrip().decode('utf8')
           pieces = line.rsplit(' ', int(nr_dim))
           word = pieces[0]
           vector = numpy.asarray([float(v) for v in pieces[1:]], dtype='f')
           nlp.vocab.set_vector(word, vector)  # add the vectors to the vocab
    """

  def train(self, df):
    """
    Same as the below build model function but uses a dataframe instead
    of paths to lyrics
    """
    self.target_dict = dict()
    self.labels = df.Emotion.unique()

    # Read files corresponding to each available emotion
    for emotion in self.labels:
      subDf = df[df.Emotion == emotion]
      # Compute sum of each lyrics document vector norm
      count, value = 0, 0
      for index, row in subDf.iterrows():
        lyric = row['Lyric_Path']
        emotion = row['Emotion']
        with open(lyric, 'r') as lyric_file:
          doc = self.nlp(lyric_file.read()) 
          value += doc.vector_norm
          count += 1
      # Add field to the target dictionary
      self.target_dict[emotion] = value/count

    logging.info('Model successfully trained: {}'.format(self.target_dict))

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

  def predict(self, path_to_lyric):
    """
    Predict the emotion for the given lyric.
    @path_to_lyric should be a valid path to a lyric
    """
    with open(path_to_lyric, 'r') as l:
      # Build doc object
      doc = self.nlp(l.read())
      # Obtain vector's norm
      value = doc.vector_norm
      # Return the predicted class
      min_dist = float(sys.maxsize)
      label = None
      for target in self.target_dict:
        dist = abs(self.target_dict[target] - value)
        if dist < min_dist:
          label = target
          min_dist = dist
      logging.info('Prediction for {}: {}'.format(path_to_lyric, label))
      return label

  def persist_target_model(self, target_path):
    """
    Persiste the built target model in a file
    """
    if self.target_dict is None:
      logging.warning('Target model does not exist yet')
      return
    if os.path.exists(target_path):
      logging.warning('Model file already exists')
      return
    # Create the model's file
    with open(target_path, 'wb') as model_file:
      pickle.dump(self.target_dict, model_file)
      logging.info('Model successfully persisted ({})'.format(target_path))

  def load_model(self, target_path):
    """
    Load serialized model object
    """
    with open(target_path, 'rb') as model_file:
      self.target_dict = pickle.load(model_file)

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
