
# coding: utf-8

# In[1]:

# TO BE USED IN BARNACLE

#MODIFIED TO BE LESS COMPLICATED

#This file performs the parsing of 5 datasets here and runs 9 algorithms on the sample.
# The program will only use the obese/lean samples (omits overweight) and specifically works on Wu, Turnbaugh, HMP,
# Amish, and Yasunenko studies

# To use this function, call the function that loops through classifiers and creates table in other ntbk
#modify to have parameter settings at the end

#In order to run this file, you must also download Parsing_Dataset and CLassifiers_Accuracies_Script (from git repo)


# In[9]:

#import statements
import biom
import sklearn
import pandas as pd
from datetime import datetime
#import nbimporter # not sure if need to download
#from imp import reload
import Parsing_Dataset_small as parse #MAKE SURE YOU DOWNLOAD THIS FROM GIT REPO
import GridSearch_AccuraciesScript as script #MAKE SURE YOU DOWNLOAD THIS GIT REPO


# In[6]:

#reload(parse)
#reload(script)


# In[7]:

#MODIFY PATH ACCORDINGLY

path = "/projects/bariatric_surgery/"


# In[8]:

# dictionary of dictionaries to contain file names
files = {
    'turnbaugh': {},
    'wu' : {},
    'amish' : {},
    'yatsunenko' : {},
    'HMP' : {},
} 

#Turnbaugh
files['turnbaugh']['meta'] = path + "merged_bmi_mapping_final__original_study_Turnbaugh_mz_dz_twins__.txt"
files['turnbaugh']['biom'] =path + "filtered_otu_table__original_study_Turnbaugh_mz_dz_twins__.biom"

#Wu dataset
files['wu']['meta'] = path + "merged_bmi_mapping_final__original_study_COMBO_Wu__.txt"
files['wu']['biom'] = path + "filtered_otu_table__original_study_COMBO_Wu__.biom"

#Amish dataset
files['amish']['meta'] = path + "merged_bmi_mapping_final__original_study_amish_Fraser__.txt"
files['amish']['biom'] = path + "filtered_otu_table__original_study_amish_Fraser__.biom"

#Yatsunenko dataset
files['yatsunenko']['meta'] = path + "merged_bmi_mapping_final__original_study_Yatsunenko_GG__.txt"
files['yatsunenko']['biom'] = path + "filtered_otu_table__original_study_Yatsunenko_GG__.biom"

#HMP dataset
files['HMP']['meta'] = path + "merged_bmi_mapping_final__original_study_HMP__.txt"
files['HMP']['biom'] = path + "filtered_otu_table__original_study_HMP__.biom"

#files


# In[37]:

#run on each study

#for study in files:
 #   isAmish = False
  #  if(study == 'amish'):
   #     isAmish = True
   # X, y = parse.parse_dataset_X(files[study]['meta'], files[study]['biom'], isAmish)
   # dataframe = script.setupAndRun(study, X, y, 5)

X, y = parse.parse_dataset_X(files['HMP']['meta'], files['HMP']['biom'], False)
dataframe = script.gridSearch('HMP', X, y, 2)
   
    
#finished all studies


# In[38]:

dataframe.to_csv('accuracytable' + str(datetime.now()) + '.csv')
# %timeit list(reversed(range(1,1000)))


# In[ ]:



