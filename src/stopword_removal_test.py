from __future__ import print_function

import spacy
import os
import sys

import pandas as pd

from utils.datasets import load_dataset_from_path

if len(sys.argv) < 2:
  print('ERROR. Please provide the dataset directory as first argument')
  sys.exit(1)

def remove_stopwords(doc):
  tks = list(filter(lambda tk: not tk.is_stop, doc))
  return spacy.tokens.Doc(nlp.vocab, words=[tk.text for tk in tks])

# For this script we will use a simple spacy vocabulary
# because we just need to see what happens when we 
# delete stopwords
nlp = spacy.load('en')

paths = load_dataset_from_path(sys.argv[1])['Lyric_Path'].as_matrix()

# Build a dataframe with the following schema:
# <song, words, words_after_stopwords_removal, percentage_change>
rows = list()

for path in paths:
  with open(path, 'r') as f:
    doc = nlp(f.read())
    n_words_before = len(doc)
    doc = remove_stopwords(doc)
    n_words_after = len(doc)
    perc = (n_words_before - n_words_after) / n_words_before * 100
    row = (path, n_words_before, n_words_after, perc)
    rows.append(row)

# Create a dataframe with the found information
df = pd.DataFrame(rows, columns=['Lyric', 'Word_Count', 
                  'Word_Count_After', 'Percentage_Change'])

# Print some statistics
percs = [ 25, 30, 40, 50, 60, 75 ]
print('Percentage of change in lyrics after removing stopwords:')
it = enumerate(percs)
plt_data = list()
for (i, perc) in it:
  if i == 0: 
    count = len(df[df['Percentage_Change'] < perc])
    print(' - < {}: {}'.format(perc, count))
    plt_data.append(('< {}'.format(perc), count))
  elif i == len(percs) - 1:
    count = len(df[df['Percentage_Change'] >= perc])
    print(' - >= {}: {}'.format(perc, count))
    plt_data.append(('>= {}'.format(perc), count))
  else:
    prev_p = percs[i-1]
    count = len(df[(df['Percentage_Change'] >= prev_p) & (df['Percentage_Change'] < perc)])
    print(' - between {} and {}: {}'.format(prev_p, perc, count))
    plt_data.append(('>= {} and < {}'.format(prev_p, perc, count), count))

pltDf = pd.DataFrame(plt_data)
print(pltDf)

# Plot
import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.barplot(pltDf[0], pltDf[1])
ax.set_title('Percentage change in number of words after stopwords pruning')
plt.show()
