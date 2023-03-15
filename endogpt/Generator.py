#------------------------------------------------------
# Regular modules
#------------------------------------------------------
import numpy as np

import pandas as pd

import random

import requests

#------------------------------------------------------
# To avoid warnings
#------------------------------------------------------
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#------------------------------------------------------
# Specific modules
#------------------------------------------------------
import torch
#from torch import datasets

import spacy
from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM

#-------------------------
# Functions called
#--------------------------
from endogpt.Classifier import single_text
from endogpt.Preprocessor import preprocess_real



def call_generator(prompt):
    API_TOKEN = "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"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/tombrooks248/EndoGPT"

    def query(payload):
        data = json.dumps(payload)
        response = requests.request("POST", API_URL, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))

    data = query(
        {
            "inputs": prompt,
            "options":{"use_cache":False, "wait_for_model":True},
            "parameters": {
                             'min_length' :30,
                             'max_length' :120,
                             'temperature' :1.0,
                             'num_return_sequences' :1,
                             'do_sample': True
                            },
        }
    )
    return data

def generate_findings(sample):

    sample['gen_findings'] = sample.prompt.map(call_generator)
    sample['gen_findings'] = sample.gen_findings.map(lambda x: x[0]['generated_text'].split("FINDINGS:")[1])

    return sample

def calculate_appearance(string):
    data = real['findings'].copy()
    total_count = data.str.count(string).sum()

    return total_count

nlp = spacy.load('en_core_web_sm')

findings_diversity = 0.6

def cosine_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)

    return doc1.similarity(doc2)

def test_similarity(text, real_findings):
    similarity = [cosine_similarity(text, real_text) for real_text in real_findings]

    return (max(similarity) - findings_diversity)

def scored_gen_findings(csv_path, num_gen):

    df = pd.read_csv(csv_path)
    real = preprocess_real(df)
    real['prompt'] = (real['General Practitioner'] + real['Endoscopist'] + real['Instrument']
                      + 'INDICATIONS FOR PROCEDURE:' + real['Indications'] + 'EXTENT OF EXAM:' + real['Extent of Exam'] + 'FINDINGS:')

    sample = real.sample(n = num_gen)

    sample = generate_findings(sample)
    sample['medical_report?'] = sample.gen_findings.map(single_text)
    sample["in_corpus?"] = sample.gen_findings.map(calculate_appearance)
    sample['similarity'] = sample.gen_findings.apply(test_similarity, args = (real.findings,))
    output = sample[['Indications','Extent of Exam', 'gen_findings', 'medical_report?','in_corpus?', 'similarity']]


    return output
