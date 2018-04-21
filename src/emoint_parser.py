import os
import re
import sys

import pandas as pd
import numpy as np

from utils.dataset_parsing import *
from utils.progress import *

EMOINT_BASEDIR = 'emoint'

emoint_ml_mapping = {
    'joy': 'happy',
    'anger': 'angry',
    'sadness': 'sad'
}

emoint_columns = ['ID', 'TWEET', 'EMOTION', 'INTENSITY']

def tweet_preprocess(tweet):
    # Remoags
    t = re.sub('\#[a-zA-Z0-9]+', '', tweet)
    # Replace tags like: @GaryHold -> GaryHold
    t = t.replace('@', '')
    # Remove emojis
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE)
    t = emoji_pattern.sub(r'', t)
    return t.rstrip()

if __name__ == '__main__':
    emoint = None
    # Read the dataset
    for ei_file in os.listdir(EMOINT_BASEDIR):
       df = pd.read_csv(os.path.join(EMOINT_BASEDIR, ei_file), delimiter='\t', header=None)
       df.columns = emoint_columns
       df['TWEET'] = df['TWEET'].apply(tweet_preprocess)
       df['EMOTION'] = df['EMOTION'].apply(lambda x: emoint_ml_mapping[x])
       df = df[df['INTENSITY'] > 0.5]
       if emoint is None:
           emoint = df
       else:
           emoint = emoint.append(df)
    # Store emoint to a csv file
    emoint.to_csv('datasets/emoint.csv', index=False)

    # Featurize EmoInt
    total = len(emoint)
    count = 0
    rows = list()
    for (idx, row) in emoint.iterrows():
        content = row['TWEET']
        content_doc = nlp(content)
        title_vect = np.zeros(content_doc.vector.shape)
        
        tweet = preprocess(content)
        features = feature_extraction(tweet, None)

        freq = features['frequencies'] 
        elem = (
            '_'.join(['EmoInt', str(row['ID'])]),
            'EmoInt', row['ID'], # Artist and title set just to match ML schema
            content_doc.vector, title_vect,
            features['line_count'], features['word_count'],#get_line_count(lyric), get_word_count(lyric),
            #get_slang_counts(lyric),
            features['echoisms'], features['selfish'],#get_echoisms(lyric), get_selfish_degree(lyric),
            count_duplicate_lines(content.split('\n')), features['is_title_in_lyrics'],# (row['Song'], lyric),
            features['rhymes'],#get_rhymes(lyric),
            features['verb_tenses']['present'], features['verb_tenses']['past'], features['verb_tenses']['future'], #verb_freq['present'], verb_freq['past'], verb_freq['future'],
            freq['ADJ'], freq['ADP'], freq['ADV'], freq['AUX'], freq['CONJ'], 
            freq['CCONJ'], freq['DET'], freq['INTJ'], freq['NOUN'], freq['NUM'],
            freq['PART'], freq['PRON'], freq['PROPN'], freq['PUNCT'], freq['SCONJ'],
            freq['SYM'], freq['VERB'], freq['X'], freq['SPACE'],
            row['EMOTION']
        )
        
        rows.append(elem)
        count += 1
        progress(count, total, '{}/{}'.format(count, total))
    df = pd.DataFrame(rows)
    df.columns = ['ID', 'ARTIST', 'SONG_TITLE', 'LYRICS_VECTOR', 'TITLE_VECTOR', 
        'LINE_COUNT', 'WORD_COUNT', 'ECHOISMS', 'SELFISH_DEGREE', 
        'DUPLICATE_LINES', 'IS_TITLE_IN_LYRICS', 'RHYMES', 'VERB_PRESENT', 
        'VERB_PAST', 'VERB_FUTURE', 'ADJ_FREQUENCIES', 'CONJUCTION_FREQUENCIES', 
        'ADV_FREQUENCIES', 'AUX_FREQUENCIES', 'CONJ_FREQUENCIES', 'CCONJ_FREQUENCIES', 
        'DETERMINER_FREQUENCIES', 'INTERJECTION_FREQUENCIES', 'NOUN_FREQUENCIES', 
        'NUM_FREQUENCIES', 'PART_FREQUENCIES', 'PRON_FREQUENCIES', 'PROPN_FREQUENCIES', 
        'PUNCT_FREQUENCIES', 'SCONJ_FREQUENCIES', 'SYM_FREQUENCIES', 'VERB_FREQUENCIES',
        'X_FREQUENCIES', 'SPACE_FREQUENCIES', 'EMOTION'
    ]
    
    output_path = 'datasets/emoint_featurized.csv'
    df.to_csv(output_path, index=False)
    
    sys.stdout.flush()
    print()
    print('Done! Saved into', output_path)
