import emoclassify as emoclassify
import playlist_classify as pclf

import numpy as np

# Install Spacy's language model
# Necessary for server
#from os import environ
#if environ.get('HEROKU_SERVER') is not None:
#import subprocess
#subprocess.check_call(["python", '-m', 'pip', 'install', 'https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.1.0a0/en_core_web_lg-2.1.0a0.tar.gz'])

from flask import Flask, json, request, redirect, render_template,url_for
app = Flask(__name__, static_url_path='')


@app.route('/classify-song', methods=['POST'])
def clf_song():
  data = request.get_json()
  artist = data['artist']
  title = data['title']
  y = emoclassify.classify('song' + str(hash(artist + title)), artist, title)
  if y is None:
    return app.response_class(status=404)
  res = {
    'artist': artist,
    'title': title,
    'emotion': y.tolist()
  }
  return app.response_class(
      response=json.dumps(res),
      status=200,
      mimetype='application/json'
  )

@app.route('/classify-playlist', methods=['POST'])
def clf_playlist():
  '''
  Robust classification scheme
  vect = request.get_json()['predictions']
  vect = np.array([np.array(x) for x in vect])
  pred, outliers_count, outliers = pclf.robust_classify(vect)
  outliers = [[int(y) for y in x] for x in outliers]
  resp = {
    'prediction': pred.tolist(),
    'outliers': outliers
  }
  '''
  vect = request.get_json()['predictions']
  vect = np.array([np.array(x) for x in vect])
  pred = pclf.classify(vect)
  resp = {
    'prediction': pred.tolist()
  }
  return app.response_class(
    response=json.dumps(resp),
    status=200,
    mimetype='application/json'
  )

@app.route("/")
def index():
  return app.send_static_file('index.html')
