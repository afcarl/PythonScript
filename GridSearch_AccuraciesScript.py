
# coding: utf-8

# In[1]:

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

# In[5]:

#Dictionaries that hold parameters 
paramsKNN = {
    'n_neighbors':[2,5,10]
}
paramsDecisionTrees = {
    'criterion':['gini', 'entropy'],
    'max_depth':[3,4,5,10,20],
    'min_samples_split': [2,3,4,5,6,10,15],
    'min_samples_leaf':[3,4,5,6,7,8,10,15],
    'max_features':['sqrt', 'log2'],   
}
paramsNB = {}
paramsSVM = {
    'C':[1,2, 10, 100],
    'kernel':['rbf', 'linear']
}
paramsGaussian = {}
paramsRandomForest = {
    'n_estimators': [20, 50, 100, 500, 1000, 1500, 1800, 2000],
    'max_depth':[2,3,4,5,10,100,200,1000],
    'min_samples_split':[2,3,5,6,10,15],
    'min_samples_leaf':[2,4,5,6,8,10,15],
    'max_features':['sqrt', 'log2']
}

# finish parameter values
paramsNeuralNet = {
  'hidden_layer_sizes':[(140,), (210,), (250,)],
  'solver':['lbfgs', 'adam']
}
paramsAdaboost = {
  'learning_rate': [1,2,3,5,10],
  'n_estimators': [15,20,30,50,100,500,1000],
  'base_estimator':[DecisionTreeClassifier(), RandomForestClassifier()]
}
paramsExtraTrees = {
    'n_estimators':[20,50,100,200,500,1000, 2000],
    'max_depth':[None, 20, 50, 100],
    'min_samples_split':[2,3,5,6,8,10,15],
    'min_samples_leaf': [1,2,3,4,5,6,10,15],
    'max_features':['auto', 'log2'], 
}


# In[14]:

# Dictionary of algorithms (with their parameters)

algs = collections.OrderedDict()
algs['KNN'] = paramsKNN
algs['Decision Tree'] = paramsDecisionTrees
algs['Naive Bayes'] = paramsNB
algs['SVM'] = paramsSVM
algs['Gaussian Process'] = paramsGaussian
algs['Random Forest'] = paramsRandomForest
algs['Neural Net'] = paramsNeuralNet
algs['AdaBoost'] = paramsAdaboost
algs['Extra Trees Classifier'] = paramsExtraTrees


df = pd.DataFrame()
df['classifier name'] = ['KNN', 'Decision Tree', 'Naive Bayes', 'SVM', 'Gaussian Process', 'Random Forest', 'Neural Net', 'AdaBoost', 'Extra Trees Classifier']


# In[2]:

def gridSearch(dataset_name, X, y, num_iterations):
    models = collections.OrderedDict() 
    for i in range(1, num_iterations):
        name = dataset_name + str(i)
        models['KNN'] = KNeighborsClassifier()
        models['Decision Tree'] = DecisionTreeClassifier(random_state=1)
        models['Naive Bayes'] = GaussianNB()
        models['SVM'] = SVC(random_state=1)
        models['Gaussian Process'] = GaussianProcessClassifier(random_state=1)
        models['Random Forest'] = RandomForestClassifier(n_jobs=8, random_state=1)
        models['Neural Net'] = MLPClassifier(random_state=1)
        models['AdaBoost'] = AdaBoostClassifier(random_state=1) 
        models['Extra Trees Classifier'] = ExtraTreesClassifier(n_jobs=8,random_state=1)
        run_dataset(name, X, y, models, algs) 
                      
    return df


# In[10]:

def run_dataset(dataset_name, X, y, models, algs):
    iter_range = range(1,6)
    average_accuracy = 0.0
    accuracy_list = []
    print(dataset_name)
    for (name, model), (name, alg) in zip(models.items(),algs.items()):
        print(model)
        #print(alg)
        clf = GridSearchCV(model, alg, n_jobs=8, cv=10, scoring='roc_auc')
        clf.fit(X, y)
        
        # print( best accuracy and associated params
        print(clf.best_score_)
        print(clf.best_params_)
        print('\n')	
        clf = clf.best_estimator_ 
	# take average over 5 iter
        for i in iter_range: 
            # append mean of best score
            classifier = clf
            scores = cross_val_score(classifier, X, y, cv=10, scoring='roc_auc')
            average_accuracy+=scores.mean()
        accuracy_list.append((average_accuracy/(5.0)))
        average_accuracy = 0.0
     
    se = pd.Series(accuracy_list)
    df[dataset_name] = se.values





