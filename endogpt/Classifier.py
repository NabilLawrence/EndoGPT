#------------------------------------------------------
# Regular modules
#------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
#------------------------------------------------------
# To avoid warnings
#------------------------------------------------------
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#------------------------------------------------------
# Not regular modules
#------------------------------------------------------
import datasets
import torch
from umap import UMAP
from transformers import AutoTokenizer, AutoModel
import pickle
#-----------------------------------------------------
def data_cleaning(df):
    """
    Input: DataFrame with medical reports.
    Output: DataFrame with "extent_of_exam","indications" and "findings".
    """
    def regex_hosp(string):
        hospital_reg = r"\.*Hospital:.*"
        line = re.findall(hospital_reg, string)[0]
        return  line.replace(',',':').split(":")[1]

    df["foundation_trust"] = df['text'].apply(regex_hosp)
    #-----------------------------------------------------
    def regex_hosp_num(string):
        hospital_reg = r"\.*Hospital Number.*"
        line= re.findall(hospital_reg, string)[0]
        return  line.replace(',',':').split(":")[1]
    df["hospital_num"] = df['text'].apply(regex_hosp_num)
    #-----------------------------------------------------
    def regex_GP(string):
        hospital_reg = r"\.*General Practitioner:.*"
        line = re.findall(hospital_reg, string)[0]
        retrn_string= line.replace(',',':').split(":")[1]
        if retrn_string[-1:] == "\r":
            return retrn_string[:-1]
        else:
            return retrn_string
    df["gp"] = df['text'].apply(regex_GP)
    #-----------------------------------------------------
    def regex_DOB(string):
        hospital_reg = r"\.*DOB:.*"
        line =  re.findall(hospital_reg, string)[0]
        retrn_string= line.replace(',',':').split(":")[1]
        if retrn_string[-1:] == "\r":
            return retrn_string[:-1]
        else:
            return retrn_string
    df["DOB"] = df['text'].apply(regex_DOB)
    #-----------------------------------------------------
    def regex_procedure_date(string):
        hospital_reg = r"\.*Date of procedure:.*"
        line =  re.findall(hospital_reg, string)[0]
        retrn_string =  line.split(":")[1][:-11]
        if retrn_string[-1:] == "\r":
            return retrn_string[:-1]
        else:
            return retrn_string
    df["procedure_date"] = df['text'].apply(regex_procedure_date)
    #-----------------------------------------------------
    def regex_endoscopist(string):
        hospital_reg = r"\.*Endoscopist:.*"
        line = re.findall(hospital_reg, string)[0]
        retrn_string =  line.replace(',',':').split(":")[1]
        if retrn_string[-1:] == "\r":
            return retrn_string[:-1]
        else:
            return retrn_string
    df["endoscopist"] = df['text'].apply(regex_endoscopist)
    #-----------------------------------------------------
    def regex_2nd_endoscopist(string):
        hospital_reg = r"\.*2nd Endoscopist:.*"
        line= re.findall(hospital_reg, string)[0]
        retrn_string =  line.replace(',',':').split(":")[1]
        if retrn_string[-1:] == "\r":
            return retrn_string
        else:
            return retrn_string
    df["second_endoscopist"] = df['text'].apply(regex_2nd_endoscopist)
    #-----------------------------------------------------
    def regex_medication(string):
        hospital_reg = r"\d*.\dmcg"
        retrn_string= re.findall(hospital_reg, string)[0]
        if retrn_string[-1:] == "\r":
            return float(retrn_string[:-4])
        else:
            return float(retrn_string[:-3])
    df["medications_fentynl"] = df['text'].apply(regex_medication)
    #-----------------------------------------------------
    def regex_midazolam(string):
        hospital_reg = r"\.*Midazolam.*"
        line = re.findall(hospital_reg, string)[0]
        retrn_string =  line.split()[1]
        if retrn_string[-1:] == "\r":
            return int(retrn_string[:-3])
        else:
            return int(retrn_string[:-2])
    df["midazolam"] = df['text'].apply(regex_midazolam)
    #-----------------------------------------------------
    def regex_instrument(string):
        hospital_reg = r"\.*Instrument.*"
        line = re.findall(hospital_reg, string)[0]
        retrn_string =  line.replace(',',':').split(":")[1]
        if retrn_string[-1:] == "\r":
            return retrn_string[:-1]
        else:
            return retrn_string
    df["instrument"] = df['text'].apply(regex_instrument)
    #-----------------------------------------------------
    def regex_extent(string):
        hospital_reg = r"\.*Extent of Exam:.*"
        line = re.findall(hospital_reg, string)[0]
        retrn_string =  line.replace(',',':').split(":")[1]
        if retrn_string[-1:] == "\r":
            return retrn_string[:-1]
        else:
            return retrn_string
    df["extent_of_exam"] = df['text'].apply(regex_extent)
    #-----------------------------------------------------
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
    #-----------------------------------------------------
    df["indications"] = df['text'].apply(regex_indications)

    def regex_procedure(string):
        hospital_reg = r"\.*Procedure Performed:.*"
        line = re.findall(hospital_reg, string)[0]
        retrn_string =  line.replace(',',':').split(":")[1]
        if retrn_string[-1:] == "\r":
            return retrn_string[:-1]
        else:
            return retrn_string
        print(retrn_string)
    df["procedure_performed"] = df['text'].apply(regex_procedure)
    #-----------------------------------------------------
    def regex_findings(string):
        hospital_reg = r"\.*FINDINGS:.*"
        line = re.findall(hospital_reg, string)[0][10:]
        return line
    df["findings"] = df['text'].apply(regex_findings)
    #-----------------------------------------------------
    df_extracted = df[["extent_of_exam","indications","findings"]]
    #-----------------------------------------------------
    return df_extracted

def including_reversing(df):
    """
    Input: DataFrame with findings from medical reports.
    Output: DataFrame with and findings and with reversed findings
            from medical reports.
    """
    #-----------------------------------------------------
    findings = df['findings']
    def reverse(row):
        row = row[::-1]
        return row
    #-----------------------------------------------------
    findings_reverse = findings.apply(reverse)
    #-----------------------------------------------------
    sentences = pd.concat([findings, findings_reverse])
    #-----------------------------------------------------
    sentences = sentences.to_frame()
    #-----------------------------------------------------
    sentences['label'] = 0
    sentences.reset_index(drop=True, inplace=True)
    for index in range(0,1000):
        sentences.at[index,'label']=1
    return sentences

def train_test_validation(sentences):
    """
    Input: DataFrame with text.
    Output: DatasetDict with train, validation and test Datasets.
    """
    #-----------------------------------------------------
    train, test = train_test_split(sentences,test_size=0.3,random_state=1)
    train.reset_index(drop=True)
    #-----------------------------------------------------
    train, validation = train_test_split(train,test_size=0.3,random_state=1)
    train.reset_index(drop=True)
    #-----------------------------------------------------
    train_dataset = datasets.Dataset.from_pandas(train)
    test_dataset = datasets.Dataset.from_pandas(test)
    validation_dataset = datasets.Dataset.from_pandas(validation)

    #-----------------------------------------------------
    train_dataset = train_dataset.remove_columns(["__index_level_0__"])
    test_dataset = test_dataset.remove_columns(["__index_level_0__"])
    validation_dataset = validation_dataset.remove_columns(["__index_level_0__"])
    #-----------------------------------------------------
    Dict = datasets.DatasetDict({"train":train_dataset,"test":test_dataset,"validation":validation_dataset})
    return Dict


def classifier_fake_real(csv_file_fake,csv_file_real):
    """
    Inputs:
    csv_file_fake: csv file with fake medical reports
    csv_file_real: csv file with real medical reports
    (both files stored in data folder)
    Outputs:
    feature_matrix_list: List with feature matrix (X)
    to be used in a classifier.
    """
    #-----------------------------------------------------
    # Including fake findings to the dataset
    #-----------------------------------------------------
    path='../data/'
    #------------------------------------------------
    fake = pd.read_csv(path+csv_file_fake)
    #------------------------------------------------
    fake=fake.rename(columns={"out":"text"})
    #------------------------------------------------
    fakedf = data_cleaning(fake)
    #-----------------------------------------------------
    print('FAKE DATA LOADED ')
    #-----------------------------------------------------
    # Including real findings to the dataset
    #-----------------------------------------------------
    real = pd.read_csv(path+csv_file_real)
    real=real.rename(columns={"out":"text"})
    realdf = data_cleaning(real)
    realfindings = realdf[['findings']]
    realfindings['label'] = 1
    #-----------------------------------------------------
    print('REAL DATA LOADED ')
    #-----------------------------------------------------
    fakefindings = fakedf[['findings']]
    fakefindings['label'] = 0
    sentences = pd.concat([fakefindings, realfindings])
    #-----------------------------------------------------
    # Creating a dictionary with train, test, validation datasets
    #-----------------------------------------------------
    Dict_datasets = train_test_validation(sentences)
    #-----------------------------------------------------
    print('DICTIONARY OF DATASETS CREATED')
    #-----------------------------------------------------
    #Load tokenizer and model
    #-----------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    #-----------------------------------------------------
    model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1", num_labels=2)
    #-----------------------------------------------------
    print('TOKENIZER AND MODEL LOADED')
    #--------------------------------------------------------------------
    # define function to tokenize the datasets
    #--------------------------------------------------------------------
    def tokenize(data):
        return tokenizer(data["findings"], padding=True, truncation=True,max_length=5)
    #--------------------------------------------------------------------
    # tokenize datasets
    #--------------------------------------------------------------------
    Dict_datasets_encoded = Dict_datasets.map(tokenize, batched=True, batch_size=None)
    #-----------------------------------------------------
    print('DICTIONARY OF DATASETS TOKENIZED')
    #--------------------------------------------------------------------
    # Formating to torch
    #--------------------------------------------------------------------
    Dict_datasets_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    #-----------------------------------------------------
    print('DICTIONARY OF DATASETS FORMATIZED')
    #--------------------------------------------------------------------
    # Extracting hidden states
    #--------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def extract_hidden_states(batch):
        # Place model inputs on the GPU
        inputs = {k:v.to(device) for k,v in batch.items()
                if k in tokenizer.model_input_names}
        # Extract last hidden states
        with torch.no_grad():
            last_hidden_state = model(**inputs).last_hidden_state
        # Return vector for [CLS] token
        return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}
    dataset_hidden = Dict_datasets_encoded.map(extract_hidden_states, batched=True)
    #-----------------------------------------------------
    print('HIDDEN STATES EXTRACTED')
    #--------------------------------------------------------------------
    # Creating a feature matrix
    #--------------------------------------------------------------------
    X_train = np.array(dataset_hidden["train"]["hidden_state"])
    X_valid = np.array(dataset_hidden["validation"]["hidden_state"])
    X_test = np.array(dataset_hidden["test"]["hidden_state"])
    y_train = np.array(dataset_hidden["train"]["label"])
    y_valid = np.array(dataset_hidden["validation"]["label"])
    y_test = np.array(dataset_hidden["test"]["label"])
    #--------------------------------------------------------------------
    feature_matrix_list = [X_train,X_valid,X_test,y_train,y_valid,y_test]
    #--------------------------------------------------------------------
    # Training a classifier
    #--------------------------------------------------------------------
    lr_clf = LogisticRegression(max_iter=100)
    lr_clf.fit(X_train, y_train)
    #-----------------------------------------------------
    print('CLASSIFIER MODEL TRAINED')
    #--------------------------------------------------------------------
    score=lr_clf.score(X_valid, y_valid)
    #-----------------------------------------------------
    print('SCORE:', score)
    #--------------------------------------------------------------------
    y_preds = lr_clf.predict(X_test)
    cm = confusion_matrix(y_test, y_preds, normalize="true")
    #-----------------------------------------------------
    print('CONFUSION MATRIX:', cm)
    #-----------------------------------------------------
    print('CLASSIFIER EVALUATED')
    #-----------------------------------------------------
    return feature_matrix_list


def single_text(custom_sentence):
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1", num_labels=2)
    #-----------------------------------------------------
    inputs = tokenizer(custom_sentence, return_tensors="pt")
    #-----------------------------------------------------
    labels = ['Real medical report','Fake medical report']
    #-----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
            outputs = model(**inputs)
    #outputs.last_hidden_state.shape
    #-----------------------------------------------------
    preds_prob = loaded_model.predict_proba(np.array(outputs.last_hidden_state[:,0].cpu().numpy()))
    return preds_prob[0][0]
