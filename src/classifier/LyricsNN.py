"""Load vectors for a language trained using fastText
https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
Compatible with: spaCy v2.0.0+
"""
import plac
import numpy

import spacy
from spacy.language import Language

import sys

class LyricsNN:
  def __init__(self):
    
    pass

  def build_lang(self, vec_path):
    # Let's use some pretrained model for now
    self.nlp = spacy.load('en_core_web_md')

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

  def build_model(self, target_dict):
    """
    Build our model which is a dictionary where the key is the target label
    and the value is the vector value associated to that target
    """
    pass

  def predict(self, path_to_lyric):
    with open(path_to_lyric, 'r') as l:
      # Build doc object
      doc = nlp(l.read()
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
      return label

@plac.annotations(
    vectors_loc=("Path to pre-trained vector model", "positional", None, str))
def main(vectors_loc, lang=None):
    if lang is None:
        nlp = Language()
    else:
        # create empty language class – this is required if you're planning to
        # save the model to disk and load it back later (models always need a
        # "lang" setting). Use 'xx' for blank multi-language class.
        nlp = spacy.blank(lang)
       # test the vectors and similarity
    text = 'class colspan'
    doc = nlp(text)
    print(text, doc[0].similarity(doc[1]))


if __name__ == '__main__':
    plac.call(main)
