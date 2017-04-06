
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
    'splitter':['best'],
    'criterion':['gini', 'entropy'],
    'min_samples_leaf':[3,4,5,6,7,8],
    'max_features': ['sqrt'],
    'max_depth':[3,4,5],
    'min_samples_split':[2,3,4,5]    
}
paramsNB = {}
paramsSVM = {
    'C':[1,2]
}
paramsGaussian = {}
paramsRandomForest = {
    'max_depth': [2,3,4,5],
    'min_samples_split': [2,3,5,6],
    'min_samples_leaf':[2,4,5],
    'n_estimators': [1500,1800]
}

# finish parameter values
paramsNeuralNet = {
  'hidden_layer_sizes':[(140,), (210,), (250,)],
  'solver':['lbfgs', 'adam']
}
paramsAdaboost = {}
paramsExtraTrees = {}


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
#df['classifier name'] = ['KNN', 'Decision Tree', 'Naive Bayes', 'SVM', 'Gaussian Process', 'Random Forest']


# In[2]:

def gridSearch(dataset_name, X, y, num_iterations):
    models = collections.OrderedDict() 
    for i in range(1, num_iterations):
        name = dataset_name + str(i)
        models['KNN'] = KNeighborsClassifier()
        models['Decision Tree'] = DecisionTreeClassifier()
        models['Naive Bayes'] = GaussianNB()
        models['SVM'] = SVC()
        models['Gaussian Process'] = GaussianProcessClassifier()
        models['Random Forest'] = RandomForestClassifier()
        models['Neural Net'] = MLPClassifier()
        models['AdaBoost'] = AdaBoostClassifier() 
        models['Extra Trees Classifier'] = ExtraTreesClassifier()
        run_dataset(name, X, y, models, algs) 
                      
    return df


# In[10]:

def run_dataset(dataset_name, X, y, models, algs):
    iter_range = range(1,6)
    average_accuracy = 0.0
    accuracy_list = []
    print(dataset_name)
    # for name, model in models:
    for (name, model), (name, alg) in zip(models.items(),algs.items()):
        print(model)
        print(alg)
        clf = GridSearchCV(model, alg, cv=10, scoring='roc_auc')
        clf.fit(X, y)
        
        # print( best accuracy and associated params
        print(clf.best_score_)
        print(clf.best_params_)
        print('\n')	
        #best_params = clf.best_estimator_.get_params()
            
        # append mean of best score
        #accuracy_list.append(cross_val_score(clf, X, y, cv=10, scoring='roc_auc').mean())
        accuracy_list.append(clf.best_score_)        

    se = pd.Series(accuracy_list)
    df[dataset_name] = se.values


# In[ ]:



