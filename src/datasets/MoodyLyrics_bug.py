import pandas as pd

moodyLyricsDF = pd.read_csv('./MoodyLyrics4Q.csv')

duplicatedCheck = moodyLyricsDF.groupby(['Artist','Title']).agg('count')
duplicatedRows = duplicatedCheck [duplicatedCheck['Index']>1]
duplicatedRows.reset_index(level=['Artist','Title'], inplace=True)

rows = list()

for (idx, row) in duplicatedRows.iterrows():
    artist, song = row['Artist'], row['Title']
    emotions = moodyLyricsDF[(moodyLyricsDF['Artist'] == artist) & (moodyLyricsDF['Title'] == song)]['Mood']
    emotions = emotions.as_matrix().tolist()
    indexes = moodyLyricsDF[(moodyLyricsDF['Artist'] == artist) & (moodyLyricsDF['Title'] == song)]['Index']
    indexes = indexes.as_matrix().tolist()
    rows.append((
        artist, song, row['Index'], emotions, indexes
    ))

bugReport = pd.DataFrame(rows, columns=['Artist', 'Title', 'Duplicate_Count', 'Emotions', 'MooyLyrics_Indexes'])

bugReport.to_csv('moodylyrics_bug_report.csv', index=False)
