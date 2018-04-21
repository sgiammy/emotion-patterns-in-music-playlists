import argparse
import json
import os
import csv
import shutil

import lyricwikia
import pandas as pd
from progress.bar import Bar


class LyricsManager:
  def __init__(self, inputDatasetFolder, outputFolder, moodyLyrics):
    self.inputDatasetFolder = inputDatasetFolder
    self.outputFolder = outputFolder
    self.moodyLyricsPath = moodyLyrics

    # Create output folder if it doesn't exist
    if not os.path.exists(self.outputFolder):
        os.makedirs(self.outputFolder)

    # This is a CLEAR folders!
    # if os.path.exists(self.outputFolder) and shutil.rmtree(self.outputFolder):
    #   os.rmdir(self.outputFolder)
    # os.mkdir(self.outputFolder)

    # Read MoodyLyrics
    self.moody_lyrics = pd.read_csv(self.moodyLyricsPath, delimiter=";")

  def get_lyric(self, spid):
    lpath = os.path.join(self.outputFolder, spid)
    if os.path.isfile(lpath):
      with open(lpath, 'r') as f:
        return f.read()

    # Download the song
    song = self.spotify[spid]
    artist, title = song['artist_name'], song['track_name']
    try:
        lyrics = lyricwikia.get_lyrics(artist, title)
    except lyricwikia.LyricsNotFound:
        return None

    with open(os.path.join(self.outputFolder, spid), 'w') as f:
        f.write(lyrics)
    return lyrics

  def build_ds(self):
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
                                        (self.moody_lyrics['Title'] == track['track_name'])
                                        ]['Mood']
            if emotion is not None and len(emotion) > 0:
              self.spotify[track_uri]['emotion'] = emotion.values[0]

  def retrieve_lyrics(self):
    self.spotify = dict()
    total = 0
    lyriced = 0

    if os.path.isfile('./lyrics.csv'):
        cache = [x.rstrip() for x in open('./lyrics.csv')]
        for line in cache:
            [track_uri, done] = line.split(';')
            total += 1
            lyriced += int(done)
            self.spotify[track_uri] = {'track_uri': track_uri, 'done': done}

    list_files = sorted(os.listdir(self.inputDatasetFolder))
    num_playlists = len(list_files)*1000

    for ds in list_files:
        with open(os.path.join(self.inputDatasetFolder, ds), 'r') as f:
            spotify_data = json.loads(f.read())
            for id_p, playlist in enumerate(spotify_data['playlists']):
                bar = Bar('Playlist %d/%d' % (id_p, num_playlists), max=len(playlist['tracks']))
                for track in playlist['tracks']:
                    bar.next()
                    track_uri = track['track_uri']
                    if track_uri in self.spotify:
                        continue
                    self.spotify[track_uri] = track
                    total += 1
                    lyr = self.get_lyric(track_uri)
                    found = (0 if lyr is None else 1)
                    lyriced += found

                    with open('./stats.txt', 'w') as stats:
                        stats.write('Last file %s' % ds)
                        stats.write('Total %d | Lyrics %d' % (total, lyriced))

                    with open('./lyrics.csv', 'a+') as table:
                        table.write('%s;%d\n' % (track_uri, found))

                bar.finish()

    print('Total %d | Lyrics %d' % (total, lyriced))



if __name__ == '__main__':
  # CLI arguments parser
  parser = argparse.ArgumentParser(description='Download and handle lyrics for songs in the the Spotify playlist dataset')
  parser.add_argument('-i', '--inputDataset', type=str, help='Spotify playlist dataset folder', required=True)
  parser.add_argument('-d', '--lyric',  type=str, help='Spotify ID of the song of which we want the lyrics')
  parser.add_argument('-o', '--outputFolder', type=str, help='Where to store lyrics', required=True)
  parser.add_argument('-m', '--materializePath', type=str, help='Where to store Spotify songs data structure once it is built')
  parser.add_argument('-l', '--dsFile', type=str, help='Load Spotify songs datastructure (if already existing)')
  parser.add_argument('-ml', '--moodyLyrics', type=str, help='MoodyLyrics dataset path for emotion tagging', required=True)

  args = parser.parse_args()

  # Use LyricsManager class
  lyrics = LyricsManager(args.inputDataset, args.outputFolder, args.moodyLyrics)
  lyrics.retrieve_lyrics()
# lyrics.build_ds(args.inputDataset)
  # l = lyrics.get_lyric('spotify:track:0UaMYEvWZi0ZqiDOoHU3YI')
  # l = lyrics.get_lyric('spotify:track:5MxNLUsfh7uzROypsoO5qe')
  # l = lyrics.get_lyric('spotify:track:0UaMYEvWZi0ZqiDOoHU3YI')
