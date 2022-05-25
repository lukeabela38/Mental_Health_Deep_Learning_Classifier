import re
import string
import nltk
nltk.download('stopwords', quiet = True)
from nltk.corpus import stopwords

def detokenize(tokens):
    text = ' '.join(tokens)
    return text

def preprocess_corpus(tokens):
  tokens = [t.lower() for t in tokens] # make lowercase

  stop = set(stopwords.words('english'))
  tokens = [t for t in tokens if t not in stop] # remove stop words
  tokens = list(filter(None, tokens)) # get rid of extra spaces

  text = detokenize(tokens)

  pattern = r"[{}]".format(string.punctuation) 
  text = re.sub(pattern, "", text)  # remove puntucation 
  text = re.sub(r'\b\d+\b', '', text) # only accept text
  
  tokens = re.split(r"\s+",text) #split the words into array
  tokens = list(filter(None, tokens)) # get rid of extra spaces
  
  return tokens