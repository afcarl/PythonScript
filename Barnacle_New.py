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

# In[3]:

#import statements
import biom
import sklearn
import pandas as pd
#import nbimporter # not sure if need to download
#from imp import reload
import Parsing_Dataset as parse #MAKE SURE YOU DOWNLOAD THIS FROM GIT REPO
import TimingAccuracies as script #MAKE SURE YOU DOWNLOAD THIS GIT REPO
from datetime import datetime

#reload(parse)
#reload(script)

# In[8]:

#MODIFY PATH ACCORDINGLY

path = "/projects/bariatric_surgery/"


# In[9]:

# dictionary of dictionaries to contain file names
files = {
    'turnbaugh': {},
    'wu' : {},
    'amish' : {},
    'yatsunenko' : {},
    'HMP' : {},    
} 
newfiles = {
    'new': {}
}

#Turnbaugh
files['turnbaugh']['meta'] = path + "merged_bmi_mapping_final__original_study_Turnbaugh_mz_dz_twins__.txt"
files['turnbaugh']['biom'] = path + "Turnbaugh.biom"

#Wu dataset
files['wu']['meta'] = path + "merged_bmi_mapping_final__original_study_COMBO_Wu__.txt"
files['wu']['biom'] = path + "Wu.biom"

#Amish dataset
files['amish']['meta'] = path + "merged_bmi_mapping_final__original_study_amish_Fraser__.txt"
files['amish']['biom'] = path + "Amish.biom"

#Yatsunenko dataset
files['yatsunenko']['meta'] = path + "merged_bmi_mapping_final__original_study_Yatsunenko_GG__.txt"
files['yatsunenko']['biom'] = path + "Yat.biom"

#HMP dataset
files['HMP']['meta'] = path + "merged_bmi_mapping_final__original_study_HMP__.txt"
files['HMP']['biom'] = path + "HMP.biom"

#new dataset
newfiles['new']['meta'] = path + "metadata_newstudy.txt"
newfiles['new']['biom'] = path + "newstudy.biom"

# In[10]:

#run on each study

#for study in files:
    #isAmish = False
    #if(study == 'amish'):
	#isAmish = True
X, y = parse.parse_dataset_X(files['amish']['meta'], files['amish']['biom'], True)
dataframe = script.gridSearch('amish', X, y, 2)
	
#finished all studies
dataframe.to_csv('accuracytable' + str(datetime.now()) + '.csv')

# In[8]:

# %timeit list(reversed(range(1,1000)))

