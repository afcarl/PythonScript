
# coding: utf-8

# In[1]:

# GridSearch for Wu and Turnbaugh


# In[9]:

#import statements
import biom
import sklearn
import pandas as pd
#import nbimporter # not sure if need to download
from imp import reload
import Parsing_Dataset as parse #MAKE SURE YOU DOWNLOAD THIS FROM GIT REPO
import Grid_Search_Accuracies_Script as script #MAKE SURE YOU DOWNLOAD THIS GIT REPO


# In[6]:

#MODIFY PATH ACCORDINGLY

path = "/projects/bariatric_surgery/"


# In[7]:

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
files['turnbaugh']['biom'] = path + "filtered_otu_table__original_study_Turnbaugh_mz_dz_twins__.biom"

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


# In[8]:

#run on each study

for study in files:
    X, y = parse.parse_dataset_X(files[study]['meta'], files[study]['biom'], False)
    dataframe = script.gridSearch(study, X, y, 2)
    dataframe
    
#finished all studies

