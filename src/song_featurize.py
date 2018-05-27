import lyricwikia
import os
import sys

import utils.dataset_parsing as dp

import utils.sentiment_analysis as sa
from utils.progress import progress

import numpy as np

LYRICS_PATH = './datasets/silver_standard/lyrics'

if not os.path.exists(LYRICS_PATH):
    os.path.makedirs(LYRICS_PATH)

LOG_FILE = '.'.join([sys.argv[0], 'log'])
try:
  os.remove(os.path.join('.', LOG_FILE))
except OSError:
  # Log file did not exists...not too bad
  pass

def err(msg):
  with open(os.path.join('.', LOG_FILE), 'a') as log:
    log.write(msg)
    log.write('\n')

COPYRIGTH_ERROR = 'Unfortunately, we are not licensed to display the full lyrics for this song at the moment. Hopefully we will be able to in the future. Until then... how about a random page?'

def download_lyric(artist, title, output_path):
    '''
    Download the lyric given artist and title
    '''
    try:
        lyric = lyricwikia.get_lyrics(artist, title)
        if lyric == COPYRITH_ERROR:
            err_msg = 'Copyright error for {}, {}'.format(artist, title)
            err(err_msg)
            raise Exception(err_msg)
        with open(output_path, 'w') as sfile:
            sfile.write(lyric)
    except lyricwikia.LyricsNotFound:
        err('Unable to download {}, {}'.format(artist, title))
        raise Exception('Could not download {}, {}'.format(artist, title))

def feature_selection(feature_vector):
    '''
    Given a full feature vector for a certain lyric, 
    this function returns just the features used
    while classifying emotions in our models
    '''
    full_feature_list = ['ID', 'ARTIST', 'SONG_TITLE', 'LYRICS_VECTOR', 'TITLE_VECTOR', 
        'LINE_COUNT', 'WORD_COUNT', 'ECHOISMS', 'SELFISH_DEGREE', 
        'DUPLICATE_LINES', 'IS_TITLE_IN_LYRICS', 'RHYMES', 'VERB_PRESENT', 
        'VERB_PAST', 'VERB_FUTURE', 'ADJ_FREQUENCIES', 'CONJUCTION_FREQUENCIES', 
        'ADV_FREQUENCIES', 'AUX_FREQUENCIES', 'CONJ_FREQUENCIES', 'CCONJ_FREQUENCIES', 
        'DETERMINER_FREQUENCIES', 'INTERJECTION_FREQUENCIES', 'NOUN_FREQUENCIES', 
        'NUM_FREQUENCIES', 'PART_FREQUENCIES', 'PRON_FREQUENCIES', 'PROPN_FREQUENCIES', 
        'PUNCT_FREQUENCIES', 'SCONJ_FREQUENCIES', 'SYM_FREQUENCIES', 'VERB_FREQUENCIES',
        'X_FREQUENCIES', 'SPACE_FREQUENCIES', 
        'SENTIMENT', 'SUBJECTIVITY']

    # Define a list of the features we want to select
    selected_features = ['LYRICS_VECTOR', 'ECHOISMS', 'DUPLICATE_LINES', 'IS_TITLE_IN_LYRICS', 
        'VERB_PRESENT', 'VERB_PAST', 'VERB_FUTURE', 'ADJ_FREQUENCIES', 'PUNCT_FREQUENCIES',
        'SENTIMENT', 'SUBJECTIVITY']
    
    # Perform feature selection
    return [feature_vector[full_feature_list.index(f)] for f in selected_features]

def vectorize(feature_vector):
    '''
    Given a feature vector, flatten it into a 1D array
    '''
    return np.hstack(feature_vector)

def preprocess_features(feature_vector, scaler):
    '''
    Applies feature preprocessing techniques, e.g. scalind data
    '''
    return scaler.transform(feature_vector.reshape(1,-1))

def featurize(sid, artist, title):
    '''
    Download lyric into LYRICS_PATH and generate feature for the given song.
    The feature tuple has the following schema:
        'ID', 'ARTIST', 'SONG_TITLE', 'LYRICS_VECTOR', 'TITLE_VECTOR', 
        'LINE_COUNT', 'WORD_COUNT', 'ECHOISMS', 'SELFISH_DEGREE', 
        'DUPLICATE_LINES', 'IS_TITLE_IN_LYRICS', 'RHYMES', 'VERB_PRESENT', 
        'VERB_PAST', 'VERB_FUTURE', 'ADJ_FREQUENCIES', 'CONJUCTION_FREQUENCIES', 
        'ADV_FREQUENCIES', 'AUX_FREQUENCIES', 'CONJ_FREQUENCIES', 'CCONJ_FREQUENCIES', 
        'DETERMINER_FREQUENCIES', 'INTERJECTION_FREQUENCIES', 'NOUN_FREQUENCIES', 
        'NUM_FREQUENCIES', 'PART_FREQUENCIES', 'PRON_FREQUENCIES', 'PROPN_FREQUENCIES', 
        'PUNCT_FREQUENCIES', 'SCONJ_FREQUENCIES', 'SYM_FREQUENCIES', 'VERB_FREQUENCIES',
        'X_FREQUENCIES', 'SPACE_FREQUENCIES', 
        'SENTIMENT', 'SUBJECTIVITY'
    '''
    
    lyric_path_name = os.path.join(LYRICS_PATH, str(sid))

    # If the lyric file does not exist, download it
    if not os.path.lexists(lyric_path_name):
        try:
            download_lyric(artist, title, lyric_path_name)        
        except Exception:
            return None

    # Read lyric file and parse it
    with open(lyric_path_name, 'r') as f:
        # Read lyric content
        content = f.read()
       
        # Preprocess content
        content_prep = dp.preprocess(content)

        # Generate SpaCy docs for both lyric and title
        lyric_doc = dp.nlp('\n'.join(content_prep))
        title_doc = dp.nlp(title)
        
        # Extract features
        features = dp.feature_extraction(content_prep, title)
        frequencies = features['frequencies'] 
        sentiment = sa.analyse(content)

        # Return extracted features
        return (
              sid,
              artist, title,
              lyric_doc.vector, title_doc.vector,
              features['line_count'], features['word_count'],
              features['echoisms'], features['selfish'],
              dp.count_duplicate_lines(content_prep), features['is_title_in_lyrics'],
              features['rhymes'],
              features['verb_tenses']['present'], features['verb_tenses']['past'], features['verb_tenses']['future'], 
              frequencies['ADJ'], frequencies['ADP'], frequencies['ADV'], frequencies['AUX'], frequencies['CONJ'], 
              frequencies['CCONJ'], frequencies['DET'], frequencies['INTJ'], frequencies['NOUN'], frequencies['NUM'],
              frequencies['PART'], frequencies['PRON'], frequencies['PROPN'], frequencies['PUNCT'], frequencies['SCONJ'],
              frequencies['SYM'], frequencies['VERB'], frequencies['X'], frequencies['SPACE'],
              sentiment[0], sentiment[1],
        )
