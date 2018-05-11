import pandas as pd

from sklearn.preprocessing import StandardScaler
from utils.dataset_parsing import *


from keras.models import load_model

def classify(lyrics, title, modelpath='emodetect.h5'):
    model = load_model(modelpath)

    # Featurize lyrics
    lyric_doc = nlp(lyrics)
    lyric = preprocess(lyrics)
    features = feature_extraction(lyric, title)

    freq = features['frequencies'] 
    sentiment = sa.analyse(content)
    elem = (
        lyric_doc.vector,
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
        sentiment[0], sentiment[1]
    )

    columns = ['LYRICS_VECTOR',
        'LINE_COUNT', 'WORD_COUNT', 'ECHOISMS', 'SELFISH_DEGREE', 
        'DUPLICATE_LINES', 'IS_TITLE_IN_LYRICS', 'RHYMES', 'VERB_PRESENT', 
        'VERB_PAST', 'VERB_FUTURE', 'ADJ_FREQUENCIES', 'CONJUCTION_FREQUENCIES', 
        'ADV_FREQUENCIES', 'AUX_FREQUENCIES', 'CONJ_FREQUENCIES', 'CCONJ_FREQUENCIES', 
        'DETERMINER_FREQUENCIES', 'INTERJECTION_FREQUENCIES', 'NOUN_FREQUENCIES', 
        'NUM_FREQUENCIES', 'PART_FREQUENCIES', 'PRON_FREQUENCIES', 'PROPN_FREQUENCIES', 
        'PUNCT_FREQUENCIES', 'SCONJ_FREQUENCIES', 'SYM_FREQUENCIES', 'VERB_FREQUENCIES',
        'X_FREQUENCIES', 'SPACE_FREQUENCIES', 'SENTIMENT', 'SUBJECTIVITY'
    ]
    df = pd.DataFrame(data=rows,columns=columns)

    # Feature selection
    selected_columns = [
        'LYRICS_VECTOR',
        'WORD_COUNT', 'ECHOISMS', 'SELFISH_DEGREE', 
        'DUPLICATE_LINES', 'IS_TITLE_IN_LYRICS', 'VERB_PRESENT', 
        'VERB_PAST', 'VERB_FUTURE', 'ADJ_FREQUENCIES',
        'PUNCT_FREQUENCIES',
        'SENTIMENT', 'SUBJECTIVITY'
    ]
    df = df[selected_columns]

    # Turn into numerical features
    X_vect = adjust(df)
    sc = StandardScaler()
    X_vect_scaled = sc.fit_transform(X_vect)

    # Return peredictions
    return classifier.predict(X_vect_scaled)
