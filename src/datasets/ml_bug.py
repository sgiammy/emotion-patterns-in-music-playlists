import pandas as pd

moodyLyricsDF = pd.read_csv('./moodylyrics_raw.csv')
duplicatedCheck = moodyLyricsDF.groupby(['Artist','Song']).size().reset_index(name='count')
duplicatedRows = duplicatedCheck [(duplicatedCheck ['count']>1)]

duplicatedRows.to_csv('moodylyrics_bug_report.csv', index=False)
