import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
import collections
from time import time
from operator import itemgetter

# Dictionaries that hold parameters

# focus parameters for both Random Forest and Extra Trees are the same to show comparable timing results
paramsRandomForest = {
    'n_estimators': [20, 50, 100, 500, 1000, 1500, 1800, 2000],
    'max_depth':[2,3,4,5,10,100,200,1000],
    'min_samples_split':[2,3,5,6,10,15],
    'min_samples_leaf':[2,4,5,6,8,10,15],
    'max_features':['sqrt', 'log2']
}

paramsExtraTrees = {
    'n_estimators':[20,50,100,500,1000,1500,1800, 2000],
    'max_depth':[2,3,4,5,10,100,200,1000],
    'min_samples_split':[2,3,5,6,10,15],
    'min_samples_leaf': [2,4,5,6,8,10,15],
    'max_features':['sqrt', 'log2'], 
}

algs = collections.OrderedDict()
algs['Random Forest'] = paramsRandomForest
algs['Extra Trees Classifier'] = paramsExtraTrees

df = pd.DataFrame()
df['classifier name'] = ['Random Forest', 'Extra Trees Classifier']

# run through all parameter combinations using gridSearch
def gridSearch(dataset_name, X, y, num_iterations):
    models = collections.OrderedDict() 
    for i in range(1, num_iterations):
        name = dataset_name + str(i)
        models['Random Forest'] = RandomForestClassifier(n_jobs=8, random_state=1)
        models['Extra Trees Classifier'] = ExtraTreesClassifier(n_jobs=8,random_state=1)
        run_dataset(name, X, y, models, algs) 
                      
    return df

#run through the datasets and output timing and accuracies for each algorithm
def run_dataset(dataset_name, X, y, models, algs):
    iter_range = range(1,6)
    average_accuracy = 0.0
    accuracy_list = []
    print(dataset_name)
    for (name, model), (name, alg) in zip(models.items(),algs.items()):
        print(model)
        start = time()
        clf = GridSearchCV(model, alg, n_jobs=8, cv=10, scoring='roc_auc')
        clf.fit(X, y)
        print("TIME")
        print(start - time())

        # print( best accuracy and associated params
        print(clf.best_score_)
        print(clf.best_params_)
        print('\n')

        #get the scoring results for the algorithms
        top_score = sorted(clf.grid_scores_, key=itemgetter(1), reverse=True)[:1]
        for score in top_score:
            print(np.std(score.cv_validation_scores))        
        accuracy_list.append(clf.best_score_)	
    se = pd.Series(accuracy_list)
    df[dataset_name] = se.values



