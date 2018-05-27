import json

import song_featurize as sf
import emoclassify
import playlist_classify as pclf

import numpy as np

def classify_slice(slice_path):
    '''
    Return emotion predictions for each playlist in the
    given slice

    Parameters:
    -----------
    slice_path: string
    '''

    # Return playlists
    s_slice = json.load(open(slice_path, 'r'))
    playlists = s_slice['playlists']

    # Classify each playlist: PID -> 'emotion vector'
    slice_predictions = dict()
    for p in playlists:
        pid = p['pid']
        predictions = list()
        # Classify each track in the playlist
        for track in p['tracks']:
            # Predict
            emo = emoclassify.classify(track['track_uri'], track['artist_name'], track['track_name'])
            if emo is not None:
                predictions.append(emo.reshape(4))
        predictions = np.array(predictions)
        # Classify playlist
        slice_predictions[pid] = pclf.robust_classify(predictions)

    return slice_predictions

