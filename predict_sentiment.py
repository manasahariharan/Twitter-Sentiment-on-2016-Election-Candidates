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
def decide_party(x):
    if x['false'] > x['sum']:
        return 0 
    elif x['false'] < x['sum']:
        return 1
    else:
        return 2

path = 'C:/Users/manas/OneDrive/Documents/578/project/projectdata/tagged_data/'

def train_classifier():
	data = pd.read_csv(path + 'tagged_data.csv',lineterminator='\n')
	X = data[['text']]
	y = data['pro_clinton']
	X_features = list(clean_tweets_text(list(X['text']), test = True, label = y))
	# pipeline_list = [('feat_vectorizer', DictVectorizer(dtype=np.int8, sparse=True, sort=False)), 
	# ('logistic', LogisticRegression(random_state=0,max_iter = 200, C = 0.01,solver='lbfgs'))]
	pipeline_list = [('feat_vectorizer', DictVectorizer(dtype=np.int8, sparse=True, sort=False)), 
	('logistic', SGDClassifier(verbose = False, loss = 'modified_huber',alpha = 4.641588833612782e-05,random_state = 33, 
		penalty = 'l1', class_weight = None,warm_start = False, early_stopping = True))]

	pipeline = Pipeline(pipeline_list)
	pipeline.fit(X_features,y)
	print(pipeline_list)
	return pipeline

def predict_sentiment_data(data_folder, user_data_name, pipeline, save_file):

	test_data = pd.read_csv(data_folder, usecols = ['id','text','user_screen_name'])
	print(len(test_data))
	X_test = test_data['text']
	X_test_features = list(clean_tweets_text(list(X_test), test = True))
	y_pred = pipeline.predict(X_test_features)
	print(pd.Series(y_pred).value_counts())
	test_data['pro_clinton'] = y_pred
	with open(save_file, mode='w', newline='\n', encoding = 'utf-8') as f:
		test_data.to_csv(f, line_terminator='\n', encoding='utf-8', index = False)
	test_data = test_data[['user_screen_name','pro_clinton']]
	user_counts = test_data.groupby(['user_screen_name']).agg(['count','sum'])
	user_counts.columns = ['count','sum']
	user_counts['false'] = user_counts['count']-user_counts['sum']

	user_counts['pro_clinton'] = user_counts.apply(decide_party, axis = 1)
	user_counts.to_csv(user_data_name)
	print(data_folder)
	print(user_counts['pro_clinton'].value_counts(normalize = 1))


pipeline_train = train_classifier()
# predict_sentiment_data("tweets_deb_conv.csv", 
# 	"user_counts_deb_conv.csv", pipeline_train, "tweets_deb_conv1.csv")
# predict_sentiment_data("tweets_deb1.csv", 
# 	"user_counts_deb1.csv", pipeline_train, "tweets1_deb1.csv")
# predict_sentiment_data("tweets_deb2.csv", 
# 	"user_counts_deb2.csv", pipeline_train, "tweets1_deb2.csv")
# predict_sentiment_data("tweets_deb3.csv", 
#  	"user_counts_deb3.csv", pipeline_train, "tweets1_deb3.csv",)
# predict_sentiment_data("tweets_election_day_new.csv", 
# 	"user_counts__election_day.csv", pipeline_train, "tweets1_election_day_new.csv")
# predict_sentiment_data("tweets_rep_conv.csv", 
# 	"user_counts_rep_conv.csv", pipeline_train, "tweets1_rep_conv.csv")
predict_sentiment_data("C:/Users/manas/OneDrive/Documents/578/project/projectdata/tweets_deb1/tweets_deb_conv.csv", 
	"user_counts_deb_conv.csv", pipeline_train, "C:/Users/manas/OneDrive/Documents/578/project/projectdata/tweets_deb1/tweets_deb_conv1.csv")
predict_sentiment_data("C:/Users/manas/OneDrive/Documents/578/project/projectdata/tweets_deb1/tweets_deb1.csv", 
	"user_counts_deb1.csv", pipeline_train, "C:/Users/manas/OneDrive/Documents/578/project/projectdata/tweets_deb1/tweets1_deb1.csv")
predict_sentiment_data("C:/Users/manas/OneDrive/Documents/578/project/projectdata/tweets_deb1/tweets_deb2.csv", 
	"user_counts_deb2.csv", pipeline_train, "C:/Users/manas/OneDrive/Documents/578/project/projectdata/tweets_deb1/tweets1_deb2.csv")
predict_sentiment_data("C:/Users/manas/OneDrive/Documents/578/project/projectdata/tweets_deb1/tweets_deb3.csv", 
 	"user_counts_deb3.csv", pipeline_train, "C:/Users/manas/OneDrive/Documents/578/project/projectdata/tweets_deb1/tweets1_deb3.csv",)
predict_sentiment_data("C:/Users/manas/OneDrive/Documents/578/project/projectdata/tweets_deb1/tweets_election_day_new.csv", 
	"user_counts__election_day.csv", pipeline_train, "C:/Users/manas/OneDrive/Documents/578/project/projectdata/tweets_deb1/tweets1_election_day_new.csv")
predict_sentiment_data("C:/Users/manas/OneDrive/Documents/578/project/projectdata/tweets_deb1/tweets_rep_conv.csv", 
	"user_counts_rep_conv.csv", pipeline_train, "C:/Users/manas/OneDrive/Documents/578/project/projectdata/tweets_deb1/tweets1_rep_conv.csv")
