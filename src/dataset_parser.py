import regex as re

import spacy
import pronouncing
from wiktionaryparser import WiktionaryParser

nlp = spacy.load('en_core_web_lg')

parser = WiktionaryParser()

def preprocess(doc):
  # Delete text between parentheses
  prep = doc
  re.sub("([\(\[])(.*?)([\)\]])", "", prep)
  # Split by line
  prep = prep.split('\n')
  # Delete empty lines
  prep = [ln for ln in prep if ln != '']
  return prep

def get_line_count(tokens):
  return len(tokens)

def get_word_count(tokens):
  count = 0
  for line in tokens:
    count += len(line.split(' '))
  return count

def get_rhymes(lines):
  # returns the number of rhymes in a lyric
  count = 0
  for i in range(len(lines)-1):
    words = lines[i].split()
    rhymes = pronouncing.rhymes(words[-1])
    next_line_words = lines[i+1].split()
    if next_line_words[-1] in rhymes:
      count += 1 
  return count / get_line_count(lines) 

def get_slang_counts(tokens):
  slang_counter = 0
  for tk in tokens:
    for word in tk:
      wiki_word = parser.fetch(word)
      if len(wiki_word) > 0:
        definitions = wiki_word[0]['definitions']
        if len(definitions) > 0:
          slang_counter += 'slang' in parser.fetch(word)[0]['definitions'][0]['text'].lower()
  return slang_counter / get_word_count(tokens)

def get_echoisms(tokens):
  vowels = ['a', 'e', 'i', 'o', 'u']
  # Do echoism count on a word level
  echoism_count = 0
  for line in tokens:
    doc = nlp(line)
    for i in range(len(doc) - 1):
      echoism_count += doc[i].text.lower() == doc[i+1].text.lower()
    # Count echoisms inside words e.g. yeeeeeeah
    for tk in doc:     
      for i in range(len(tk.text) - 1):
        if tk.text[i] == tk.text[i+1] and tk.text in vowels:
          echoism_count += 1
          break
  return echoism_count / get_word_count(tokens)

def get_verb_tense_frequencies(lines):
  freq = dict()
  freq['present'] = 0
  freq['past'] = 0
  verbs_no = 0

  for line in lines:
    doc = nlp(line)
    for token in doc:
      if token.pos_ == 'VERB': 
        verbs_no += 1
      if 'present' in spacy.explain(token.tag_):
        freq['present'] += 1
      elif 'past' in spacy.explain(token.tag_):
        freq['past'] += 1 

  for key, value in freq.items():
    freq[key] = value/verbs_no

  return freq 

def get_frequencies(tokens):
  freq = dict()
  for line in tokens:
    doc = nlp(line)
    for word in doc:
      if word.pos_ in freq:
        freq[word.pos_] += 1
      else:
        freq[word.pos_] = 1
  return freq

def get_selfish_degree(tokens):
  pronouns_count, i_count = 0, 0
  for line in tokens:
    doc = nlp(line)
    for tk in doc:
      if tk.tag_ == 'PRP':
        pronouns_count += 1
        if tk.text == 'I':
          i_count += 1
  return i_count / pronouns_count

def count_duplicate_lines(tokens):
  return sum([tokens.count(x) for x in list(set(tokens)) if tokens.count(x) > 1]) / get_word_count(tokens)

def is_title_in_lyrics(title, tokens):
  for tk in tokens:
    if title in tk:
      return True
  return False

f = open('ml_lyrics/sad_Warpaint_Composure', 'r')
lyric = preprocess(f.read())
print('LC:',get_line_count(lyric))
print('WC:',get_word_count(lyric))
#print('Slang:', get_slang_counts(lyric))
print('Echoisms:', get_echoisms(lyric))
print('Selfish:', get_selfish_degree(lyric))
print('Duplicates:', count_duplicate_lines(lyric))
print('Title in lyrics:', is_title_in_lyrics('Composure', lyric))

print('Rhymes:',get_rhymes(lyric))
print('Verbs:', get_verb_tense_frequencies(lyric))
print('Frequencies:', get_frequencies(lyric))
