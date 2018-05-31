import emoclassify as emoclassify
import playlist_classify as pclf

import numpy as np

from flask import Flask, json, request, redirect, render_template
app = Flask(__name__)

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
  vect = request.get_json()['predictions']
  vect = np.array([np.array(x) for x in vect])
  pred, outliers_count, outliers = pclf.robust_classify(vect)
  print(type(pred), type(outliers_count), type(outliers))
  outliers = [[int(y) for y in x] for x in outliers]
  print(type(outliers[0]), type(outliers[0][0]), outliers)
  resp = {
    'prediction': pred.tolist(),
    'outliers': outliers
  }
  return app.response_class(
    response=json.dumps(resp),
    status=200,
    mimetype='application/json'
  )

@app.route("/")
def index():
  return app.send_static_file('index.html')

