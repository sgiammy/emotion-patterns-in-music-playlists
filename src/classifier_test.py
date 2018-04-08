from classifier.LyricsNN import LyricsNN
import os

from utils.datasets import *

import logging

logging.getLogger().setLevel(logging.INFO)

# Load dataset
dataset = load_dataset_from_path('./ml_lyrics')
trainDf, testDf = split_train_validation(dataset)

# Define path where we will persist the model
model_path = os.path.join(os.getcwd(), 'emotion-model')

clf = LyricsNN()

clf.build_lang(os.path.abspath('./models/wiki-news-300d-1M.vec'))

#if os.path.exists(model_path):
  # If we have already built the model just read it
#  clf.load_model(model_path)
#else:
# Create the new model
clf.train(trainDf)
#clf.build_model('./ml_lyrics')
  
# Store the built model on disk for future speed up
clf.persist_target_model(model_path)

# Do some cross validation
acc = clf.score(testDf)
print('Obtained accuracy:', acc)
