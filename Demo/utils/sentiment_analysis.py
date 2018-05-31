import urllib.parse
import urllib.request

import json

from textblob import TextBlob

SA_URI = "http://text-processing.com/api/sentiment/"

def analyse_online(text):
    data = urllib.parse.urlencode({"text": text}).encode('utf-8')
    u = urllib.request.urlopen(SA_URI, data)
    resp = u.read()
    sa = json.loads(resp.decode('utf-8'))
    return sa

def analyse(text):
    opinion = TextBlob(text)
    sentiment = opinion.sentiment
    return (sentiment.polarity, sentiment.subjectivity)
