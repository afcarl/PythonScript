# 5/10/17 - Multiclass script
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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import collections
# In[5]:

#Dictionaries that hold parameters 
paramsKNN = {
    'estimator__n_neighbors':[2,5,10]
}
paramsDecisionTrees = {
    'estimator__criterion':['gini', 'entropy'],
    'estimator__max_depth':[5,10,20],
    'estimator__min_samples_split': [2,5,10,15],
    'estimator__min_samples_leaf':[5,10,15],
    'estimator__max_features':['sqrt'],   
}
paramsNB = {}
paramsSVM = {
    'estimator__C':[1,2, 10, 100],
    'estimator__kernel':['rbf', 'linear']
}
paramsGaussian = {}
paramsRandomForest = {
    'estimator__n_estimators': [100, 500, 1000,2000],
    'estimator__max_depth':[3,5,10,100],
    'estimator__min_samples_split':[5,10,15],
    'estimator__min_samples_leaf':[5,10,15],
    'estimator__max_features':['sqrt', 'log2']
}

paramsNeuralNet = {
  'estimator__hidden_layer_sizes':[(140,), (210,)],
  'estimator__solver':['lbfgs', 'adam']
}
paramsAdaboost = {
  'estimator__learning_rate': [1],
  'estimator__n_estimators': [50,100,500,1000],
  'estimator__base_estimator':[DecisionTreeClassifier(), RandomForestClassifier()]
}
paramsExtraTrees = {
    'estimator__n_estimators':[100,500,1000,2000],
    'estimator__max_depth':[4,10,100],
    'estimator__min_samples_split':[2,5,10,15],
    'estimator__min_samples_leaf': [2,5,10,15],
    'estimator__max_features':['sqrt', 'log2'], 
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
        models['KNN'] = OneVsRestClassifier(KNeighborsClassifier())
        models['Decision Tree'] = OneVsRestClassifier(DecisionTreeClassifier(random_state=1))
        models['Naive Bayes'] = OneVsRestClassifier(GaussianNB())
        models['SVM'] = OneVsRestClassifier(SVC(random_state=1))
        models['Gaussian Process'] = OneVsRestClassifier(GaussianProcessClassifier(random_state=1))
        models['Random Forest'] = OneVsRestClassifier(RandomForestClassifier(n_jobs=8, random_state=1))
        models['Neural Net'] = OneVsRestClassifier(MLPClassifier(random_state=1))
        models['AdaBoost'] = OneVsRestClassifier(AdaBoostClassifier(random_state=1))
        models['Extra Trees Classifier'] = OneVsRestClassifier(ExtraTreesClassifier(n_jobs=8,random_state=1))
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
	y = label_binarize(y, classes=[1,2,3])
        clf = GridSearchCV(model, alg, n_jobs=8, cv=10, scoring='accuracy')
	#start = time()
        clf.fit(X, y)
        # print( best accuracy and associated params
        print(clf.best_score_)
        print(clf.best_params_)
        print('\n')
	# print std. deviation
	top_score = sorted(clf.grid_scores_, key=itemgetter(1), reverse=True)[:1]
        for score in top_score:
            print(np.std(score.cv_validation_scores))        
        accuracy_list.append(clf.best_score_)	
        #clf = clf.best_estimator_ 
    se = pd.Series(accuracy_list)
    df[dataset_name] = se.values



