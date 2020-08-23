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
from clean_tweets_functions import clean_tweets_text
from sklearn.metrics import classification_report, confusion_matrix,f1_score,make_scorer

path = 'C:/Users/manas/OneDrive/Documents/578/project/projectdata/tagged_data/'
data = pd.read_csv(path + 'tagged_data.csv',lineterminator='\n')
X = data[['text']]
y = data['pro_clinton']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
X_train_features = list(clean_tweets_text(list(X_train['text']), test = True, label = y_train))
#X_features = list(clean_tweets_text(list(X['text'])))

pipeline_list = [('feat_vectorizer', DictVectorizer(dtype=np.int8, sparse=True, sort=False)), 
('logistic', SGDClassifier(verbose = False, loss = 'modified_huber',alpha = 4.641588833612782e-05,random_state = 33, 
     penalty = 'l1', class_weight = None,warm_start = False, early_stopping = True))]
# pipeline_list = [('feat_vectorizer', DictVectorizer(dtype=np.int8, sparse=True, sort=False)), 
# ('logistic', LogisticRegression(random_state=0,max_iter = 200, C = 0.001,solver='lbfgs'))]
pipeline = Pipeline(pipeline_list)
pipeline.fit(X_train_features,y_train)
#pipeline.fit(X_features,y)

X_test_features = list(clean_tweets_text(list(X_test['text']), test = True, label = list(y_test)))
y_pred = pipeline.predict(X_test_features)
print(classification_report(y_test, y_pred, target_names=["anti clinton","pro clinton"]))
print(confusion_matrix(y_test, y_pred, labels = [True, False]))
