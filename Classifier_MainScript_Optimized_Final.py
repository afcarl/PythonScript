
# coding: utf-8

# In[1]:

# This file performs the parsing of 5 datasets here and runs 9 algorithms on the sample.
# The program will only use the obese/lean samples (omits overweight) and specifically works on Wu, Turnbaugh, HMP,
# Amish, and Yasunenko studies

# To use this function, call the function that loops through classifiers and creates table in other ntbk
#modify to have parameter settings at the end

#In order to run this file, you must also download Parsing_Dataset and CLassifiers_Accuracies_Script (from git repo),
#and nmimporter (can find online from github)


# In[1]:

#import statements
import biom
import sklearn
import pandas as pd
import nbimporter # MUST DOWNLOAD THIS FROM ONLINE GIT (just google it and you will find it)
from imp import reload
import Parsing_Dataset as parse #MAKE SURE YOU DOWNLOAD THIS FROM GIT REPO
import Classifiers_Accuracies_Script as script #MAKE SURE YOU DOWNLOAD THIS GIT REPO
from datetime import datetime

# In[2]:

# reload scripts and their definitions
reload(parse)
reload(script)


# In[3]:

#FILE NAMES (this is the part where everyone would need to modify indiviudally)

#Turnbaugh
turnbaughMeta = "/Users/shwetakinger/Desktop/ERSP_data/merged_bmi_mapping_final__original_study_Turnbaugh_mz_dz_twins__.txt"
turnbaughBiom = "/Users/shwetakinger/Downloads/per_study_otu_tables/filtered_otu_table__original_study_Turnbaugh_mz_dz_twins__.biom"

#Wu dataset
wuMeta = "/Users/shwetakinger/Desktop/ERSP_data/merged_bmi_mapping_final__original_study_COMBO_Wu__.txt"
wuBiom = "/Users/shwetakinger/Downloads/per_study_otu_tables/filtered_otu_table__original_study_COMBO_Wu__.biom"

#Amish dataset
amishMeta = "/Users/shwetakinger/Desktop/ERSP_data/merged_bmi_mapping_final__original_study_amish_Fraser__.txt"
amishBiom = "/Users/shwetakinger/Downloads/per_study_otu_tables/filtered_otu_table__original_study_amish_Fraser__.biom"

#Yatsunenko dataset
yatsunenkoMeta = "/Users/shwetakinger/Desktop/ERSP_data/merged_bmi_mapping_final__original_study_Yatsunenko_GG__.txt"
yatsunenkoBiom = "/Users/shwetakinger/Downloads/per_study_otu_tables/filtered_otu_table__original_study_Yatsunenko_GG__.biom"

#HMP dataset
HMPMeta = "/Users/shwetakinger/Desktop/ERSP_data/merged_bmi_mapping_final__original_study_HMP__.txt"
HMPBiom = "/Users/shwetakinger/Downloads/per_study_otu_tables/filtered_otu_table__original_study_HMP__.biom"


# In[10]:

#run on Turnbaugh studies
X, y = parse.parse_dataset_X(turnbaughMeta, turnbaughBiom, False)
dataframe = script.setupAndRun('Turnbaugh', X, y, 2)
#end of Taunbaugh


# In[11]:

#run on Wu studies
X, y = parse.parse_dataset_X(wuMeta, wuBiom, False)
dataframe = script.setupAndRun('Wu', X, y, 2)

#end of Wu


# In[ ]:

#run on Amish studies
X, y = parse.parse_dataset_X(amishMeta, amishBiom, True)
dataframe = script.setupAndRun('Amish', X, y, 2)

#end of Amish


# In[ ]:

#run on Yatsunenko studies
X, y = parse.parse_dataset_X(yatsunenkoMeta, yatsunenkoBiom, False)
dataframe = script.setupAndRun('Yatsunenko', X, y, 2)

#end of Yatsunenko


# In[ ]:

#run on HMP studies
X, y = parse.parse_dataset_X(HMPMeta, HMPBiom, False)
dataframe = script.setupAndRun('HMP', X, y, 2)
dataframe.to_csv("accuracytable" + str(datetime.now()) + ".csv");
#end of HMP


# In[ ]:



