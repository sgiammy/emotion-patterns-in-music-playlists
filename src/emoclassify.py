from sklearn.externals import joblib

from keras.models import load_model

import song_featurize as sf

MODEL_PATH = 'emodetect.h5'
SCALER_PATH = 'emo_standard_scaler.pkl'
ENCODER_PATH='emo_label_encoder.pkl'

# Load the pretrained model
model = load_model(MODEL_PATH)

# Load scaler and label encoders
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

def classify(sid, artist, title, lyric_content=None):
    '''
    Generate the 4 emotions vector for the given song.

    Parameters
    ----------
    sid: string
         Spotify's track ID, used for recognizing the file in which the analyzed lyric is stored
    artist: string
    title: string
    '''
    # Featurize lyrics
    feature_vector = sf.featurize(sid, artist, title, lyric_content)
    if feature_vector is None:
        # There has been an errror e.g. unable to download the lyric
        return None
    feature_vector = sf.feature_selection(feature_vector)

    # Apply features preprocessing before classifiecation
    x = sf.vectorize(feature_vector)
    x = sf.preprocess_features(x, scaler)

    # Return peredictions
    return model.predict(x)
