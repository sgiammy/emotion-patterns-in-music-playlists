import urllib.parse
import urllib.request

import json

SA_URI = "http://text-processing.com/api/sentiment/"

def analyse(text):
    data = urllib.parse.urlencode({"text": text}).encode('utf-8')
    u = urllib.request.urlopen(SA_URI, data)
    resp = u.read()
    sa = json.loads(resp.decode('utf-8'))
    return sa
