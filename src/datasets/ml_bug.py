import pandas as pd

moodyLyricsDF = pd.read_csv('./moodylyrics_raw.csv')

duplicatedCheck = moodyLyricsDF.groupby(['Artist','Song']).agg('count')
duplicatedRows = duplicatedCheck [duplicatedCheck['Index']>1]
duplicatedRows.reset_index(level=['Artist','Song'], inplace=True)

print(moodyLyricsDF.head(2))

rows = list()

for (idx, row) in duplicatedRows.iterrows():
    artist, song = row['Artist'], row['Song']
    emotions = moodyLyricsDF[(moodyLyricsDF['Artist'] == artist) & (moodyLyricsDF['Song'] == song)]['Emotion']
    emotions = emotions.as_matrix().tolist()
    rows.append((
        artist, song, row['Index'], emotions
    ))

bugReport = pd.DataFrame(rows, columns=['Artist', 'Song', 'Duplicate_Count', 'Emotions'])

bugReport.to_csv('moodylyrics_bug_report.csv', index=False)
