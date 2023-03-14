#----------------------------------------------------------
# Regular modules
#----------------------------------------------------------
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#----------------------------------------------------------
# Visualization
#----------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#----------------------------------------------------------
# Classifiers
#----------------------------------------------------------
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
#----------------------------------------------------------
# For metrics
#----------------------------------------------------------
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#----------------------------------------------------------
# To avoid warnings
#----------------------------------------------------------
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#----------------------------------------------------------
# Not regular modules
#----------------------------------------------------------
import datasets # to create a dictionary of datasets.
import torch #The torch module provides support for multi-dimensional arrays called tensors.
#from umap import UMAP #Uniform Manifold Approximation and Projection
# is a machine learning technique for dimensionality reduction, which is commonly
# used for visualizing high-dimensional data in two or three dimensions.
#----------------------------------------------------------
# Transformers
#----------------------------------------------------------
from transformers import AutoTokenizer # to tokenize dataset of text.
from transformers import AutoModel # to export last hidden layer from the outputs of the model.
from transformers import AutoModelForSequenceClassification # to export logists from the outputs of the model.
#----------------------------------------------------------
def cleaning_real(df):
    """
    -----------------------------------------------
    Description: Cleaning the loaded dataset and separed the relevant features.
    -----------------------------------------------
        - Input: DataFrame with medical reports (Unnamed: 0, out, NA).
        - Output: DataFrame with separated features from the original out feature.
    -----------------------------------------------
    """
    #--------------------------------------------------------------------
    # List of features for regex_list
    #--------------------------------------------------------------------
    hospital_numb = r"\.*Hospital Number.*"
    hospital = r"\.*Hospital:.*"
    general_practitioner = r"\.*General Practitioner:.*"
    DOB = r"\.*DOB:.*"
    Endoscopist = r"\.*Endoscopist:.*"
    Endoscopist_2 = r"\.*2nd Endoscopist:.*"
    Instrument = r"\.*Instrument.*"
    Extent = r"\.*Extent of Exam:.*"
    Procedure = r"\.*Procedure Performed:.*"
    #--------------------------------------------------------------------
    list_features_regex = [hospital_numb,\
                           hospital,\
                           general_practitioner,\
                          DOB,\
                          Endoscopist,\
                          Endoscopist_2,\
                          Instrument,\
                          Extent,\
                          Procedure]
    #--------------------------------------------------------------------
    def regex_list(string,feature):
        """
        -----------------------------------------------
        Inputs:
            - string: All text included in the feature out for each row (str).
            - feature: feature to extract (str).
        Output:
            - retrn_string: returned string with the information of each feature.
            Example: for feature=hospital_numb retrn_string
        -----------------------------------------------
        """
        hospital_reg =  feature#r"\.*Hospital Number.*"
        line = re.findall(hospital_reg, string)[0]
        retrn_string= line.replace(',',':').split(":")[1]
        if retrn_string[-1:] == "\r":
            return retrn_string[:-1]
        else:
            return retrn_string
    #--------------------------------------------------------------------
    df["Hospital Number"] = df['out'].apply(regex_list, args=(hospital_numb,))
    df["Hospital"] = df['out'].apply(regex_list, args=(hospital,))
    df["General Practitioner"] = df['out'].apply(regex_list, args=(general_practitioner,))
    df["DOB"] = df['out'].apply(regex_list, args=(DOB,))
    df["Endoscopist"] = df['out'].apply(regex_list, args=(Endoscopist,))
    df["2nd Endoscopist"] = df['out'].apply(regex_list, args=(Endoscopist_2,))
    df["Instrument"] = df['out'].apply(regex_list, args=(Instrument,))
    df["Extent of Exam"] = df['out'].apply(regex_list, args=(Extent,))
    df["Procedure Performed"] = df['out'].apply(regex_list, args=(Procedure,))
    #--------------------------------------------------------------------
    # Date of procedure
    #--------------------------------------------------------------------
    Date_procedure = r"\.*Date of procedure:.*"
    #--------------------------------------------------------------------
    def regex_procedure_date(string):
        hospital_reg = r"\.*Date of procedure:.*"
        line =  re.findall(hospital_reg, string)[0]
        retrn_string =  line.split(":")[1][:-11]
        if retrn_string[-1:] == "\r":
            return retrn_string[:-1]
        else:
            return retrn_string
    #--------------------------------------------------------------------
    df["Date of procedure"] = df['out'].apply(regex_procedure_date)
    #--------------------------------------------------------------------
    # Medication
    #--------------------------------------------------------------------
    dmcg = r"\d*.\dmcg"
    #--------------------------------------------------------------------
    def regex_medication(string):
        hospital_reg = r"\d*.\dmcg"
        retrn_string= re.findall(hospital_reg, string)[0]
        if retrn_string[-1:] == "\r":
            return float(retrn_string[:-4])
        else:
            return float(retrn_string[:-3])
    #--------------------------------------------------------------------
    df["Medication"] = df['out'].apply(regex_medication)
    #--------------------------------------------------------------------
    # Midazolam
    #--------------------------------------------------------------------
    Midazolam = r"\.*Midazolam.*"
    #--------------------------------------------------------------------
    def regex_midazolam(string):
        hospital_reg = r"\.*Midazolam.*"
        line = re.findall(hospital_reg, string)[0]
        retrn_string =  line.split()[1]
        if retrn_string[-1:] == "\r":
            return int(retrn_string[:-3])
        else:
            return int(retrn_string[:-2])
    #--------------------------------------------------------------------
    df["Midazolam"] = df['out'].apply(regex_midazolam)
    #--------------------------------------------------------------------
    # Indications for procedure
    #--------------------------------------------------------------------
    def regex_indications(string):
        hospital_reg = r"\.*INDICATIONS FOR PROCEDURE:.*"
        line = re.findall(hospital_reg, string)[0]
        retrn_string =  line.replace(',',':').split(":")[1]
        if retrn_string[-1:] == "\r":
            retrn_string= retrn_string[:-1]
        if retrn_string[-8:] == "FINDINGS":
            return retrn_string[:-8]
        else:
            return retrn_string
    #--------------------------------------------------------------------
    df["Indications"] = df['out'].apply(regex_indications)
    #--------------------------------------------------------------------
    # Findings
    #--------------------------------------------------------------------
    def regex_findings(string):
        hospital_reg = r"\.*FINDINGS:.*"
        line = re.findall(hospital_reg, string)[0][10:]
        return line
    #--------------------------------------------------------------------
    df["findings"] = df['out'].apply(regex_findings)
    #--------------------------------------------------------------------
    return df

#--------------------------------------------------------------------
def cleaning_synthetic(synthetic):
    synthetic = synthetic[['findings']]
    synthetic['label'] = 1
    return synthetic

#--------------------------------------------------------------------
def extracting_real(df):
    """
    -----------------------------------------------
    Description: Extracting relevan features from df.
    -----------------------------------------------
     Input: df: DataFrame resulted from Cleaning.
     Output: df_extracted: DataFrame with "extent_of_exam","indications" and "findings".
    """
    #--------------------------------------------------------------------
    df_extracted = df[["General Practitioner","Endoscopist","Instrument","Extent of Exam","Indications","findings"]]
    #--------------------------------------------------------------------
    return df_extracted

#--------------------------------------------------------------------
def preprocess_real(real):
    #real = pd.read_csv(string)
    real = cleaning_real(real)
    real = extracting_real(real)
    #real.to_csv('/data/real_preprocessed.csv')
    return real

#--------------------------------------------------------------------
def preprocess_synthetic(synthetic):
    #synthetic = pd.read_csv(string)
    synthetic = cleaning_synthetic(synthetic)
    #synthetic.to_csv('/data/synthetic_preprocessed.csv')
    return synthetic
#--------------------------------------------------------------------
