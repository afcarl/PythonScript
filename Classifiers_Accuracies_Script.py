
# coding: utf-8

# In[ ]:

#This script imports all necessary classifiers and modules. A setting dict is used to store parameters
#and their initial values. 
# The setupAndRun function takes input X and y, and number of iterations to run classifiers on dataset
#A dataframe for each dataset is returned to main script. 


# In[2]:

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



# In[3]:

#Dictionary that holds parameters 
arr = [
         {'n_neighbors': 2, 'algorithm': 'auto'},
         {} ,
        {},
        {'C': 1.0, 'kernel':'linear'},
        {},
        {'n_estimators': 1000, 'max_depth': 10, 'max_features':'sqrt'},
        #hidden layer sizes (increment by 10), n_estimators, learning rate (increment by 1 each iter)
        {'hidden_layer_sizes': 150 , 'max_iter': 200},
        {'base_estimator': RandomForestClassifier(), 'n_estimators': 1000, 'learning_rate': 1.0},
        {'n_estimators': 1000, 'max_depth': 10, 'max_features': 'sqrt'}
]

df = pd.DataFrame()
df['classifier name'] = ['KNN', 'Decision Tree', 'Naive Bayes', 'SVM', 'Gaussian Process', 'Random Forest', 'Neural Net', 'AdaBoost', 'Extra Trees Classifier']


# In[1]:


def setupAndRun(dataset_name, X, y, num_iterations):
    
    for i in range(1, num_iterations):
        name = dataset_name + str(i)
        models = []
        models.append(('KNN', KNeighborsClassifier(**arr[0])))
        models.append(('Decision Tree', DecisionTreeClassifier()))
        models.append(('Naive Bayes', GaussianNB()))
        models.append(('SVM', SVC(**arr[3])))
        models.append(('Gaussian Process', GaussianProcessClassifier()))
        models.append(('Random Forest', RandomForestClassifier(**arr[5])))
        models.append(('Neural Net', MLPClassifier(**arr[6])))
        models.append(('AdaBoost', AdaBoostClassifier(**arr[7])))
        models.append(('Extra Trees Classifier', ExtraTreesClassifier(**arr[8])))
       
        run_dataset(name, X, y, models)
        
        #increment n_neighbors
        arr[0]['n_neighbors']+=2
        #increment n_estimators by 500
        arr[7]['n_estimators']+=500
        #increment hidden_layer_sizes
        arr[6]['hidden_layer_sizes']+=100
        #increment number of trees used in random forest
        arr[5]['n_estimators'] += 500
        #increment learning_rate
        arr[7]['learning_rate'] += 1.0
        
        
    return df


# In[ ]:

def run_dataset(dataset_name, X, y, models):
    iter_range = range(1,6)
    average_accuracy = 0.0
    
    accuracy_list = []
    for name, model in models:
        #print(name)
        for i in iter_range:
            classifier = model
            #print (model)
            scores = cross_val_score(classifier, X, y, cv=10, scoring='roc_auc')
            average_accuracy+=scores.mean()
        accuracy_list.append((average_accuracy/(5.0)))
        average_accuracy = 0.0
        
    se = pd.Series(accuracy_list)
    df[dataset_name] = se.values
   
    

