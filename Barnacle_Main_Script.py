# For use in Barnacle

# To use this function, call the function that loops through classifiers and creates table in other ntbk
#In order to run this file, you must also download Parsing_Dataset and GridSearch_AccuraciesScript (from git repo)

#import statements
import biom
import sklearn
import pandas as pd
import Parsing_Dataset as parse #MAKE SURE YOU DOWNLOAD THIS FROM GIT REPO
import GridSearch_AccuraciesScript as script #MAKE SURE YOU DOWNLOAD THIS GIT REPO
from datetime import datetime

# modify this path accordingly
path = "/projects/bariatric_surgery/"

# dictionary of dictionaries to contain file names (for the original 5 studies)
files = {
    'turnbaugh': {},
    'wu' : {},
    'amish' : {},
    'yatsunenko' : {},
    'HMP' : {},    
}

# used for the new study (the Ercolini study)
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


#run the script on each study
for study in files:

    #the Amish study needs additional preprocessing
    isAmish = False
    if(study == 'amish'):
	isAmish = True

    X, y = parse.parse_dataset_X(files[study]['meta'], files[study]['biom'], isAmish)
    dataframe = script.gridSearch(study, X, y, 2)
	
#finished all studies, write results to file
dataframe.to_csv('accuracytable' + str(datetime.now()) + '.csv')
