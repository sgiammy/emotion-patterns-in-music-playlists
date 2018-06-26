import song_featurize as sf
import utils.progress as progress

import pandas as pd

ML4Q_PATH = './datasets/MoodyLyrics4Q.csv'

df = pd.read_csv(ML4Q_PATH)

X = list()
total = len(df)
i = 0
for idx, row in df.iterrows():
    progress.progress(i, total, 'Parsed songs')
    point = sf.featurize(row['Index'], row['Artist'], row['Title'])
    if point is not None:
        point.append(row['Mood'])
        X.append(point)
    i += 1


df = pd.DataFrame(X, columns=[
    'ID', 'ARTIST', 'SONG_TITLE', 'LYRICS_VECTOR', 'TITLE_VECTOR', 
    'LINE_COUNT', 'WORD_COUNT', 'ECHOISMS', 'SELFISH_DEGREE', 
    'DUPLICATE_LINES', 'IS_TITLE_IN_LYRICS', 'RHYMES', 'VERB_PRESENT', 
    'VERB_PAST', 'VERB_FUTURE', 'ADJ_FREQUENCIES', 'CONJUCTION_FREQUENCIES', 
    'ADV_FREQUENCIES', 'AUX_FREQUENCIES', 'CONJ_FREQUENCIES', 'CCONJ_FREQUENCIES', 
    'DETERMINER_FREQUENCIES', 'INTERJECTION_FREQUENCIES', 'NOUN_FREQUENCIES', 
    'NUM_FREQUENCIES', 'PART_FREQUENCIES', 'PRON_FREQUENCIES', 'PROPN_FREQUENCIES', 
    'PUNCT_FREQUENCIES', 'SCONJ_FREQUENCIES', 'SYM_FREQUENCIES', 'VERB_FREQUENCIES',
    'X_FREQUENCIES', 'SPACE_FREQUENCIES', 
    'SENTIMENT', 'SUBJECTIVITY', 'EMOTION'])

df.to_csv('./datasets/MoodyLyrics4Q_featurized_fasttext.csv')
