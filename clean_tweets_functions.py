from nltk.tokenize import TweetTokenizer
from nltk import ngrams
from itertools import chain
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import TweetTokenizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.metrics import classification_report, confusion_matrix,f1_score,make_scorer
from sklearn.model_selection import KFold, GridSearchCV
import ujson as json

def get_hashtag_list(pro_clinton = True):
    if pro_clinton == True:
        with open('antitrumptags.txt') as fin:
            anti_trump = fin.readline()
        anti_trump = anti_trump.split()
        with open('prohillarytags.txt') as fin:
            pro_clinton = fin.readline()
        pro_clinton = pro_clinton.split()
        proclinton_tags = anti_trump+pro_clinton
        proclinton_tags = ['#'+tag for tag in proclinton_tags]
        return proclinton_tags
    else:
        with open('protrump.txt') as fin:
            pro_trump = fin.readline()
        pro_trump = pro_trump.split()
        with open('anticlintontags.txt') as fin:
            anti_clinton = fin.readline()
        anti_clinton = anti_clinton.split()
        anticlinton_tags = anti_clinton + pro_trump
        anticlinton_tags = ['#'+tag for tag in anticlinton_tags]
        return anticlinton_tags

proclinton_tags = get_hashtag_list(True)
anticlinton_tags = get_hashtag_list(False)
both = proclinton_tags+anticlinton_tags

def bag_of_words(words):

    return dict([(word, True) for word in words])

def bag_of_words_and_bigrams(words):
    
    bigrams = ngrams(words, 2)
    
    return bag_of_words(chain(words, bigrams))
def clean_for_test( words, label = None):
    if label == True:

        tags = proclinton_tags
        #print(proclinton_tags)
    elif label == False:
        tags = anticlinton_tags
        #print(anticlinton_tags)
    else:
        tags = both
    words = [word for word in words if word not in tags]

    return words

def clean_tweets_text(text, preserve_case=False, reduce_len=True, strip_handles=False, test = False, label = None):
    tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)

    if isinstance(text, str):
        tokens = tknzr.tokenize(text)
        if test :
            if label is None:
                tokens = clean_for_test(tokens)
            else:
                tokens = clean_for_test(tokens,label)
            

        features = bag_of_words_and_bigrams(tokens)
    elif isinstance(text, list):
        tokens = map(tknzr.tokenize, text)
        if test:
            if label is None:
                tokens = map(clean_for_test, tokens)
            else:
                tokens = map(clean_for_test, tokens, label)
                #print(list(tokens))
        features = map(bag_of_words_and_bigrams, tokens)

    return features


