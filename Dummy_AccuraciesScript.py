#Dummy accuracies
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
from sklearn.dummy import DummyClassifier
# In[5]:

# Run dummy classifier on datasets
def runClassifier(dataset_name, X, y):
    print(dataset_name)
    #prints score with most frequent label
    clf = DummyClassifier(random_state=1, strategy='most_frequent')
    print(cross_val_score(clf, X, y, cv=10).mean()) 
    print('\n')
    classifier = DummyClassifier(random_state=1, strategy='uniform')
    print(cross_val_score(classifier, X, y, cv=10).mean())
    print('\n')
    #ROC - AUC
    print(cross_val_score(clf, X, y, cv=10, scoring='roc_auc').mean())
