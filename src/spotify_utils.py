import spacy
import os
import sys
import json
import itertools
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from utils.datasets import load_dataset_from_path, split_train_validation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.dataset_parsing import *
from utils.progress import progress
import utils.sentiment_analysis as sa
import lyricwikia
import io

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

def build_ann(input_size,optimizer='adam'):
    classifier = Sequential()
    classifier.add(Dense(units = 60, kernel_initializer = 'random_normal', activation = 'sigmoid', input_dim = input_size))
    classifier.add(Dropout(0.5))

    classifier.add(Dense(units = 60, kernel_initializer = 'random_normal', activation = 'sigmoid', input_dim = input_size))
    classifier.add(Dropout(0.5))

    classifier.add(Dense(units = 4, kernel_initializer = 'random_normal', activation = 'softmax'))

    classifier.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier


def read_spotify_json(path):
    with open(path) as json_data:
        file = json.load(json_data)
        rows = list()
        for playlist in file['playlists']:
            for track in playlist['tracks']:
                line = list()
                line = [playlist['pid'], playlist['name'], track['track_uri'], track['artist_name'], track['track_name']]
                #print(line)
                rows.append(line)

    #print(rows[0])
    columns = ['PlaylistPid','PlaylistName','TrackUri', 'ArtistName', 'TrackName']
    #print(columns)
    df = pd.DataFrame(data = rows, columns = columns)

    output_path = './datasets/Spotify1stSlice.csv'
    df.to_csv(output_path, index=False)
    return df

def featurize(fname,row):
    with io.open(fname, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
        lyric_doc = nlp(content)
        lyric = preprocess(content)
        features = feature_extraction(lyric, row['TrackName'])

        freq = features['frequencies'] 
        sentiment = sa.analyse(content)
        elem = (
            row['PlaylistPid'],row['PlaylistName'], row['TrackUri'], row['ArtistName'], row['TrackName'],
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
    return elem

def download_and_featurize(df, lyrics_dir,output_path, intersectionDf=None, intersectionLog=None):
    rows = list() 
    total = len(df)
    count = 0

    needIntersection = intersectionDf is not None and intersectionLog is not None
    if needIntersection:
        intersectionFile = open(intersectionLog, 'a+')
    
    # LYRICS DOWNLOAD
    if not os.path.lexists(lyrics_dir):
        os.mkdir(lyrics_dir)

    for idx, row in df.iterrows():
        fname = '.'.join([lyrics_dir+'/'+row['TrackUri'],'txt'])
        if not os.path.lexists(fname):
            try:
                lyrics = lyricwikia.get_lyrics(row['ArtistName'], row['TrackName'])
                if needIntersection:
                    # Update intersection log
                    if ((intersectionDf.ARTIST == row['ArtistName']) & (intersectionDf.SONG_TITLE == row['TrackName'])).any():
                        intersectionFile.write('{}, {}'.format(row['ArtistName'], row['TrackName']))
                with open('.'.join([lyrics_dir+'/'+row['TrackUri'],'txt']), 'w') as f:
                    f.write(lyrics)
            except lyricwikia.LyricsNotFound:
                lyrics = None
                continue
        
        # FEATURIZATION PART   
        if os.path.lexists(fname):
            elem = featurize(fname,row)
        
        rows.append(elem)
        count += 1
        progress(count, total, '{}/{}'.format(count, total))

    # DATASET CREATION
    columns = ['PlaylistPid','PlaylistName','TrackUri', 'ArtistName', 'TrackName','LyricVector','lineCount','wordCount',
              'Echoisms','Selfish','DuplicateLines','IsTitleInLyrics','Rhymes','VerbPresent','VerbPast','VerbFuture',
              'ADJ','ADP','ADV','AUX','CONJ','CCONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','PUNCT','SCONJ','SYM',
              'VERB','X','SPACE','Sentiment','Subjectivity']
    new_df = pd.DataFrame(data=rows,columns=columns)

    new_df.to_csv(output_path, index=False)
    print()
    print('Done! Dataset written:', output_path)
    return 

def adjust(df):
    X_vect = list()
    for (i, row) in df.iterrows():
        sub_list = list()
        for field in row:
            if type(field) == str:
                field = field[1:-1].split()
                sub_list += [float(x.replace('\n','')) for x in field]
            else:
                sub_list.append(field)
        X_vect.append(np.array(sub_list))
    X_vect = np.array(X_vect)
    return X_vect

def train_logreg(X, y):
    clf = LogisticRegression(penalty='l2', dual=False, C=0.15, 
                             solver='newton-cg', multi_class='multinomial', random_state=0)
    clf.fit(X, y)
    return clf

def train_and_predict(X_train,X_test, y_train, y_test):
    encoder = LabelEncoder()
    y_train_nn = np_utils.to_categorical(encoder.fit_transform(y_train))
    #Feature Scaling
    sc = StandardScaler()
    X_train_nn = sc.fit_transform(X_train)
    X_test_nn = sc.transform(X_test)
    classifier = build_ann(input_size=X_train_nn.shape[1])
    classifier.fit(X_train_nn, y_train_nn, batch_size = 256, epochs = 100, verbose=0)
    #Predicting the Test set results
    y_pred = classifier.predict(X_test_nn,verbose=0)
    y_pred_index = np.argmax(y_pred,axis=1)
    #Validating the results
    y_nn_pred = np_utils.to_categorical(encoder.transform(y_test))
    
    cm = confusion_matrix(y_pred_index, y_nn_pred.argmax(axis=1))
    accuracy = (sum([cm[i,i] for i in range(len(cm))])) / len(y_nn_pred)
    return (classifier, sc, accuracy,encoder)
