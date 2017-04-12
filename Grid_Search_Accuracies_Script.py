
# coding: utf-8

# In[14]:

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
from sklearn import svm


from sklearn.grid_search import GridSearchCV


# In[16]:

#Dictionaries that hold parameters 
#paramsRandomForest = {
#   'max_depth': [],
#    'min_samples_split': [2,3,5,6],
#    'min_samples_leaf':[2,4,5],
#    'n_estimators': [1500,1800]
#}

paramsRandomForest = {
    'n_estimators': [2, 5, 20, 100, 1000, 2000, 3000],
    'max_features': [None, 'auto', 'log2'],
    #'max_depth': [None, 50, 100, 200, 1000, 2000],
    'n_jobs': [1, 5, 8],
    #'min_samples_split': [2, 5, 10, 15],
    #'min_samples_leaf': [1, 5, 10, 15],
    #'max_leaf_nodes': [None, 5, 10, 30, 50]
}

paramsDecisionTrees = {
    'splitter': ['best', 'random'],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [1, 5, 10, 15],
    'min_samples_leaf': [1, 5, 10, 15],
    'max_leaf_nodes': [None, 5, 10, 30, 50],   
}

paramsExtraTrees = {
    'n_estimators': [2, 5, 20, 50, 100, 500, 1000, 2000, 3000],
    'max_features': [None, 'auto', 'log2'],
    'max_depth': [None, 50, 100, 200, 1000, 2000],
    'n_jobs': [8],
    'min_samples_split': [1, 5, 10, 15],
    'min_samples_leaf': [1, 5, 10, 15],
    'max_leaf_nodes': [None, 5, 10, 15, 30, 50]
}

paramsAdaboost = {
    'algorithm': ['SAMME', 'SAMME.R'],
    'learning_rate': [0.3, 0.5, 0.8, 1., 1.5, 2, 3, 5, 10],
    'n_estimators': [2, 5, 10, 15, 20, 30, 50, 100, 200, 500, 1000, 2000, 3000],
    'base_estimator': [DecisionTreeClassifier, RandomForestClassifier, SVC],
}


paramsNeuralNet = {
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'batch_size': ['auto', 5, 10, 20, 30, 50, 80, 100, 200, 500],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'bax_iter': [5, 10, 20, 50, 100, 500, 1000],
    'shuffle': [True, False],
    'momentum': [0, 0.1, 0.5, 0.8, 1],
    'warm_start': [True, False],
    'nesterovs_momentum': [True, False],
    'validation_fraction': [0, 0.1, 0.5, 0.8, 1],
    'early_stopping': [True, False]
}
paramsSVM = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    'degree': [2, 3, 5, 10],
    'coef0': [0, 1, 2, 5, 10],
    'probability': [True, False],
    'shrinking': [True, False]
}


# In[17]:

# Dictionary of algorithms (with their parameters)

algs = {
    'randomForest': paramsRandomForest,
    'decisionTrees': paramsDecisionTrees,
    'extraTrees': paramsExtraTrees,
    # 'KNN': paramsKNN,
    'adaboost': paramsAdaboost,
    'neuralNet': paramsNeuralNet,
    'SVM': paramsSVM,
    #'naiveBayes': {},
    #'gaussian': {},
} 


df = pd.DataFrame()
df['classifier name'] = ['Decision Tree', 'SVM', 'Random Forest', 'Neural Net', 'AdaBoost', 'Extra Trees Classifier']


# In[20]:

def gridSearch(dataset_name, X, y, num_iterations):
    
    for i in range(1, num_iterations):
        name = dataset_name + str(i)
        models = []
        #models.append(('KNN', KNeighborsClassifier()
        models.append(('Decision Tree', DecisionTreeClassifier())
        #models.append(('Naive Bayes', GaussianNB())
        models.append(('SVM', SVC()))
        #models.append(('Gaussian Process', GaussianProcessClassifier()))
        models.append(('Random Forest', RandomForestClassifier()))
        models.append(('Neural Net', MLPClassifier()))
        models.append(('AdaBoost', AdaBoostClassifier()))
        models.append(('Extra Trees Classifier', ExtraTreesClassifier()))
       
        run_dataset(name, X, y, models, algs) 
                      
    return df


# In[21]:

def run_dataset(dataset_name, X, y, models, algs):
    #iter_range = range(1,6)
    average_accuracy = 0.0
    
    accuracy_list = []
    # for name, model in models:
    for (name, model), (alg, params) in zip(models.items(), algs.items()):
        clf = GridSearchCV(model, params, cv=10, scoring='roc_auc')
        clf.fit(X, y)
        
        # print( best accuracy and associated params
        print(clf.best_score_)
        print(clf.best_params_)
            
        # append mean of best score
        accuracy_list.append(cross_val_score(clf, X, y, cv=10, scoring='roc_auc').mean())
        
    se = pd.Series(accuracy_list)
    df[dataset_name] = se.values

