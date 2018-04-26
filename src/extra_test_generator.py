from utils.dataset_parsing import *

songs = [
  ('Bobby McFerrin', 'Don\'t Worry, Be Happy', 'happy'),
  ('Queen', 'Don\'t Stop me Now', 'happy'),
  ('Pharrell Williams', 'Happy', 'happy'),
  ('The Monkees', 'I\'m a believer', 'happy'),
  
  ('R.E.M.', 'Everybody Hurts', 'sad'),
  ('Adele', 'Someone Like You', 'sad'),
  ('Pink Floyd', 'Wish you were here', 'sad'),
  ('Johnny Cash', 'Hurt', 'sad'),
  ('Nirvana', 'Smells like teen spirit', 'sad'),
  
  ('Rage Against the Machine', 'Killing in the name', 'angry'),
  ('Kanye West', 'Stronger', 'angry'),
  ('Smash Mouth', 'All Star', 'angry'),
  ('Bloodhound Gang', 'The Ballad of Chasey Lain', 'angry'),
  
  ('Blur', 'Song 2', 'relaxed') # I'm not quite confident about this labeling
]

if __name__ == '__main__':
  import pandas as pd
  import numpy as np
  import lyricwikia
  import os
  import io
  import threading
  from utils.progress import progress

  # Split the dataset in 4 parts and run 4 parallel threads for doing this
  rows = list() # Rows of our dataset
  total = len(songs)
  count = 0
  for song in songs:
    content = lyricwikia.get_lyrics(song[0], song[1])
    lyric_doc = nlp(content)
    title_doc = nlp(song[1])

    lyric = preprocess(content)
    features = feature_extraction(lyric, song[1])

    freq = features['frequencies'] 
    elem = (
        song[0], song[1],
        lyric_doc.vector, title_doc.vector,
        features['line_count'], features['word_count'],#get_line_count(lyric), get_word_count(lyric),
#get_slang_counts(lyric),
        features['echoisms'], features['selfish'],#get_echoisms(lyric), get_selfish_degree(lyric),
        count_duplicate_lines(lyric), features['is_title_in_lyrics'],# (row['Song'], lyric),
        features['rhymes'],#get_rhymes(lyric),
        features['verb_tenses']['present'], features['verb_tenses']['past'], features['verb_tenses']['future'], #verb_freq['present'], verb_freq['past'], verb_freq['future'],
        freq['ADJ'], freq['ADP'], freq['ADV'], freq['AUX'], freq['CONJ'], 
        freq['CCONJ'], freq['DET'], freq['INTJ'], freq['NOUN'], freq['NUM'],
        freq['PART'], freq['PRON'], freq['PROPN'], freq['PUNCT'], freq['SCONJ'],
        freq['SYM'], freq['VERB'], freq['X'], freq['SPACE'],
        song[2]
    )

    rows.append(elem)
    count += 1
    progress(count, total, '{}/{}'.format(count, total))

  df = pd.DataFrame(rows)

  output_path = 'datasets/extra_test.csv'
  df.to_csv(output_path)
  print()
  print('Done! Dataset written:', output_path)
