import json
import argparse
import sys
import os
import shutil

import pandas as pd

import lyricwikia

class LyricsManager:
  def __init__(self, inputDatasetFolder, outputFolder, moodyLyrics):
    self.inputDatasetFolder = inputDatasetFolder
    self.outputFolder = outputFolder
    self.moodyLyricsPath = moodyLyrics

    # Create output folder if it doesn't exist
    if os.path.exists(self.outputFolder) and shutil.rmtree(self.outputFolder):
      os.rmdir(self.outputFolder)
    os.mkdir(self.outputFolder)

    # Read MoodyLyrics
    self.moody_lyrics = pd.read_csv(self.moodyLyricsPath)

  def get_lyric(self, spid):
    lpath = os.path.join(self.outputFolder, spid)
    if os.path.isfile(lpath):
      with open(lpath, 'r') as f:
        return f.read()
    else:
      # Download the song
      song = self.spotify[spid]
      artist, title = song['artist_name'], song['track_name']
      lyrics = lyricwikia.get_lyrics(artist, title)
      with open(os.path.join(self.outputFolder, spid), 'w') as f:
        f.write(lyrics)
      return lyrics

  def build_ds(self, dataset_path):
    self.spotify = dict()
    for ds in os.listdir(self.inputDatasetFolder):
      with open(os.path.join(self.inputDatasetFolder, ds), 'r') as f:
        spotify_data = json.loads(f.read())
        for playlist in spotify_data['playlists']:
          playlist_id = playlist['pid']
          for track in playlist['tracks']:
            track_uri = track['track_uri']
            if track_uri not in self.spotify:   
              self.spotify[track_uri] = track
              self.spotify[track_uri]['playlists'] = [playlist_id]
            else:
              self.spotify[track_uri]['playlists'].append(playlist_id)
            # Let's see if we have an emotion for it
            emotion = self.moody_lyrics[(self.moody_lyrics['Artist'] == track['artist_name']) &
                                        (self.moody_lyrics['Song'] == track['track_name'])
                                      ]['Emotion']
            if emotion is not None and len(emotion) > 0:
              self.spotify[track_uri]['emotion'] = emotion.values[0]

if __name__ == '__main__':
  # CLI arguments parser
  parser = argparse.ArgumentParser(description='Download and handle lyrics for songs in the the Spotify playlist dataset')
  parser.add_argument('-i', '--inputDataset', type=str, help='Spotify playlist dataset folder')
  parser.add_argument('-d', '--lyric',  type=str, help='Spotify ID of the song of which we want the lyrics')
  parser.add_argument('-o', '--outputFolder', type=str, help='Where to store lyrics')
  parser.add_argument('-m', '--materializePath', type=str, help='Where to store Spotify songs data structure once it is built')
  parser.add_argument('-l', '--dsFile', type=str, help='Load Spotify songs datastructure (if already existing)')
  parser.add_argument('-ml', '--moodyLyrics', type=str, help='MoodyLyrics dataset path for emotion tagging')

  args = parser.parse_args()

  # Use LyricsManager class
  lyrics = LyricsManager(args.inputDataset, args.outputFolder, args.moodyLyrics)
  lyrics.build_ds(args.inputDataset)
  l = lyrics.get_lyric('spotify:track:0UaMYEvWZi0ZqiDOoHU3YI')
  l = lyrics.get_lyric('spotify:track:5MxNLUsfh7uzROypsoO5qe')
  l = lyrics.get_lyric('spotify:track:0UaMYEvWZi0ZqiDOoHU3YI')
