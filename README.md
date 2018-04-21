# emotion-patterns-in-music-playlists

Before running any script, please install dependencies

    pip install -r requirements.txt

## lyrics_downloader.py

This script simply downloads the lyric of the songs contained
in any CSV file formatted in a similar way to MoodyLyrics dataset
files (`Index`,`Artist`,`Song`,`Emotion`). For a better understanding
of the options wich can be used with this script, please run:

`python3 lyrics_downloader.py -h`

From the `src` folder. If you want to test it with the dataset provided in
this repository, please run:

`python3 lyrics_downloader.py -i datasets/moodylyrics_raw.csv -o LYRICS_DIR -s`

The script will download the lyrics for the songs listed into the `src/datasets/moodylyrics.csv`
into the `src/LYRICS_DIR` folder. Each file will follow this name convention:
`EMOTION_ARTIST_SONG-TITLE`.


## LyricsManager.py

Map Spotify track URIs to LyricsWiki, and download the lyrics in a folder.

- Download and unzip [MoodyLyrics](http://softeng.polito.it/erion/MoodyLyrics.zip)
    - Convert `ml_raw.xlsx` to `ml_raw.csv` (to be better documented)
- Download and unzip the MPD
- run it


        python src/LyricsManager.py -ml /Users/pasquale/Desktop/MoodyLyrics/ml_raw.csv -i /Users/pasquale/Desktop/mpd.v1/data/ -o ./tmp/

