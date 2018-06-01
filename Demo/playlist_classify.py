import numpy as np

from scipy.stats import iqr

def robust_classify(playlist_vect):
    '''
    Given a playlist of emotions, return a vector using robust 
    classification approach (with outliers removal) as described
    in the report

    Parameters
    ----------
    playlist_vect: numpy.ndarray
                   A 2D vector with shape (num_tracks, 4) containing the emotional
                   classification of each of the track of the playlist
    '''
    if len(playlist_vect) < 4 and len(playlist_vect) > 0:
        # Avoid doing this classification if there are too few elements
        return np.mean(playlist_vect, axis=0), 0, None
    if len(playlist_vect) == 0:
        return np.zeros(4), 0, None

    prediction = np.zeros(4)
    # Iterate over each emotion and look for its outliers
    outliers_count = 0
    
    outliers = list()
    for i in np.arange(4):
        emo = playlist_vect[:, i]
        
        # Calculate the interquartile range for our data
        IQR = iqr(emo)

        # Compute first and third quartile
        q1 = np.percentile(emo, .25)
        q3 = np.percentile(emo, .75)

        # Add 1.5 x (IQR) to the third quartile. Any number greater than this is a suspected outlier.
        upper = (IQR * 1.5) + q3

        # Subtract 1.5 x (IQR) from the first quartile. Any number less than this is a suspected outlier.
        low = q1 - (IQR * 1.5)

        # Filter out outliers
        upper_outliers = emo > upper
        outliers += [[j, i] for j, val in enumerate(upper_outliers) if val]
        outliers_count += upper_outliers.sum()

        low_outliers = emo < low
        outliers += [[j, i] for j, val in enumerate(low_outliers) if val]
        outliers_count += upper_outliers.sum()
       
        prediction[i] = np.mean([x for j, x in enumerate(emo)
         if not upper_outliers[j] and not low_outliers[j]])

    return prediction, outliers_count, outliers
