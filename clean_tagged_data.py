import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from clean_tweets_functions import clean_tweets_text
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix,f1_score,make_scorer
from sklearn.model_selection import KFold, GridSearchCV
best_params_file = "best_params_file.json"

path = 'C:/Users/manas/OneDrive/Documents/578/project/projectdata/tagged_data/'
data = pd.read_csv(path + 'tagged_data.csv',lineterminator='\n')
X = data[['text']]
y = data['pro_clinton']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
X_train_features = list(clean_tweets_text(list(X_train['text']), test = True, label = y_train))
# dictvec= DictVectorizer(dtype=np.int8, sparse=True, sort=False)
# X_train_matrix = dictvec.fit_transform(X_train_features)
# print(X_train_matrix.shape)
pipeline_list = [('feat_vectorizer', DictVectorizer(dtype=np.int8, sparse=True, sort=False)), 
('logistic', SGDClassifier(verbose = False, random_state = 33,
	warm_start = False, early_stopping = True))]
pipeline = Pipeline(pipeline_list)

param_grid = {
    'logistic__alpha' : np.logspace(-1,-7, num=10),
    'logistic__penalty': ['l1','l2'],
    'logistic__loss':['modified_huber','log'],
    'logistic__class_weight':[None,'balanced']
}
scorer = make_scorer(f1_score)
kfold = KFold(n_splits=7, shuffle=True, random_state=34)
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=kfold,scoring=scorer, verbose = 4, 
	return_train_score = 'warn')
print("\nPerforming grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
print(param_grid)

grid_search.fit(X_train_features, y_train)
print("cv_results")
print(grid_search.cv_results_)
print("best estimator")
print(grid_search.best_estimator_)


print("\nBest score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters_np = grid_search.best_estimator_.get_params()
print(best_parameters_np)

best_parameters = {'logistic__loss': 'log', 'logistic__penalty': 'l2',
                                'logistic__max_iter': 300, 'logistic__alpha': 0.01,
                              'logistic__class_weight' : None}
        
# update and print best parameters
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters_np[param_name]))
    
    # convert numpy dtypes to python types
    if hasattr(best_parameters_np[param_name], 'item'):
        best_parameters[param_name] = best_parameters_np[param_name].item()
    else:
        best_parameters[param_name] = best_parameters_np[param_name]
        
# save best params to JSON file
with open(best_params_file, 'w') as fopen:
    json.dump(best_parameters, fopen)
X_test_features = list(clean_tweets_text(list(X_test['text']), test = True, label = list(y_test)))
y_pred = grid_search.predict(X_test_features)
print(classification_report(y_test, y_pred, target_names=["anti clinton","pro clinton"]))
print(confusion_matrix(y_test, y_pred, labels = [True, False]))
