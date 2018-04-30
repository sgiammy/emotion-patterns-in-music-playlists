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
from utils.datasets import load_dataset_from_path, split_train_validation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.dataset_parsing import *
from utils.progress import progress
import utils.sentiment_analysis as sa
import lyricwikia
import io
from spotify_utils import *
import os

'''
Train the classifier
'''
# MoodyLyrics Preprocessing
dataset = pd.read_csv('./datasets/moodylyrics_featurized.csv')
selected_columns = [
   'LYRICS_VECTOR',
   'WORD_COUNT', 'ECHOISMS', 'SELFISH_DEGREE', 
   'DUPLICATE_LINES', 'IS_TITLE_IN_LYRICS', 'VERB_PRESENT', 
   'VERB_PAST', 'VERB_FUTURE', 'ADJ_FREQUENCIES',
   'PUNCT_FREQUENCIES','SENTIMENT','SUBJECTIVITY',
   'EMOTION'
]
dataset = dataset[selected_columns]
train_df = dataset.drop(['EMOTION'], axis=1)
# X and Y vect creation
X_vect = adjust(train_df)
y = dataset['EMOTION'].as_matrix().T
# Train,test split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size = 0.2, random_state = 0)
# Neural network: model definition, training,prediction and validation
classifier,sc,accuracy,encoder = train_and_predict(X_train, X_test, y_train, y_test)
print('Accuracy: %0.2f' % (accuracy*100))

featurize = False

SPOTIFY_SLICES = [ 'dataset/mpd.slice.0-999.json' ] #os.listdir('spotify_dataset') # 

for (idx, slic) in enumerate(SPOTIFY_SLICES):

    output_path = 'datasets/Spotify_slice{:04}_featurized.csv'.format(idx)

    if featurize:
        '''
        We first read the json files extracted from the Spotify dataset
        We aim to: 
            1. Read the json files and save into a pandas dataframe the following information: 
               <PlaylistPid, PlaylistName, TrackUri, ArtistName, TrackName>
            2. Download the song lyrics and featurize the created dataset following the features selection described in the previous notebook
            3. Train our classificator with the MoodyLyrics dataset (4 emotions: happy, sad, angry, relaxed)
            4. Predict the emotion of each track in the spotify dataframe and add 4 columns: 
               <PlaylistPid, PlaylistName, TrackUri, ArtistName, TrackName, Happy_CL, Sad_CL, Angry_CL, Relaxed_CL>
               where CL stands for confidence level. 
            5. Once we classified each song in a playlist, we sum up the rows for each emotion confidence level.
               We then create a new dataframe: 
               <PlaylistPid, PlaylistName,Happy_CL, Sad_CL, Angry_CL, Relaxed_CL>

        '''

        '''
        PART 1: Read the json files and save into a pandas dataframe the following information: 
               <PlaylistPid, PlaylistName, TrackUri, ArtistName, TrackName>
        '''
        df = read_spotify_json(slic)
        '''
        END PART 1 
        '''


        '''
        PART 2:	Download the song lyrics and featurize the created dataset following the features 
                selection described in the previous notebook
        '''
        lyrics_dir = './spotify_lyrics'
        
        new_df = download_and_featurize(df, lyrics_dir,output_path)
        '''
        END PART 2
        '''

    '''
    PART 3: Train our classificator with the MoodyLyrics dataset (4 emotions: happy, sad, angry, relaxed)
    '''
    # Moved before the for

    '''
    PART 4:  Predict the emotion of each track in the spotify dataframe and add 4 columns: 
           <PlaylistPid, PlaylistName, TrackUri, ArtistName, TrackName, Happy_CL, Sad_CL, Angry_CL, Relaxed_CL>
           where CL stands for confidence level. 
    '''
    new_df = pd.read_csv('datasets/Spotify1stSlice_featurized.csv')#output_path)
    selected_columns = ['PlaylistPid', 'PlaylistName', 'TrackUri', 'ArtistName', 'TrackName',
           'LyricVector', 'wordCount', 'Echoisms', 'Selfish',
           'DuplicateLines', 'IsTitleInLyrics', 'VerbPresent',
           'VerbPast', 'VerbFuture', 'ADJ', 'PUNCT', 'Sentiment', 'Subjectivity'
    ]

    tmp_df = new_df[selected_columns]
    tmp_df = tmp_df.drop(['PlaylistPid', 'PlaylistName', 'TrackUri', 'ArtistName', 'TrackName'], axis=1)


    X_vect = adjust(tmp_df)
    #X_vect_scaled = sc.transform(X_vect)
    y_pred = classifier.predict(X_vect, verbose=0)
    #y_pred_label = [list(zip(y_pred[i], emotions_label)) for i in range(len(y_pred))]
    #y_pred_ord = [sorted(y_pred_label[i], key=lambda x:  x[0], reverse=True) for i in range(len(y_pred_label))]
    emotion_labels = encoder.inverse_transform([0,1,2,3])
    classificationDf = pd.DataFrame(data=y_pred,columns=emotion_labels)
    finalDf = pd.concat([new_df, classificationDf],axis=1)
    output_path = './datasets/spotify_lyrics_slice{:04}_classified.csv'.format(idx)
    finalDf.to_csv(output_path,index=False)

    '''
    PART 5: Classify playlists 
    '''
    clfDf = finalDf.groupby(by='PlaylistPid').agg({'happy': 'mean', 'sad': 'mean', 'angry': 'mean', 'relaxed': 'mean'})
    clfDf.to_csv('./datasets/classified_playlists_slice{:04}.csv'.format(idx))
