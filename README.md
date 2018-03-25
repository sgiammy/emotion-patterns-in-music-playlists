# emotion-patterns-in-music-playlists

# lyrics_downloader.py

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
