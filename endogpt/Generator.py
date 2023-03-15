#------------------------------------------------------
# Regular modules
#------------------------------------------------------
import numpy as np
import pandas as pd
import random
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
from torch import datasets

import spacy
from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM

#-------------------------
# Functions called
#--------------------------

from endogpt.Preprocessor import preprocess_real
from endogpt.Classifier import single_text


def scored_gen_findings(string, num_gen):


    df = pd.read_csv(string)#('data/real.csv') # loads data and creates prompt
    real = preprocess_real(df)
    real['prompt'] = (real['General Practitioner'] + real['Endoscopist'] + real['Instrument']
                  + 'INDICATIONS FOR PROCEDURE:' + real['Indications'] + 'EXTENT OF EXAM:' + real['Extent of Exam'] + 'FINDINGS:')

    tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
    model = AutoModelForCausalLM.from_pretrained("tombrooks248/EndoGPT")

    nlp = spacy.load('en_core_web_sm') # Load pre-trained word embedding model

    findings_diversity = 0.6

    def generate_findings(df,num = num_gen):    # generates findings

        generator = pipeline('text-generation', model=model, tokenizer=tokenizer,
                     min_length = 30,
                     max_length = 120,
                     temperature = 1.0,
                     num_return_sequences = 1,
                     do_sample=True,
                     #top_k = 3
                     #top_p = 0.6
                    )

        sample = df.sample(n = num_gen)
        sample['gen_findings'] = sample.prompt.map(generator)
        sample['gen_findings'] = sample.gen_findings.map(lambda x: x[0]['generated_text'].split("FINDINGS:")[1])

        return sample

    def calculate_appearance(string): # simlarity 1: has the findings already appeared in the original dataset

        data = real['findings'].copy() # str.count() method to count given string appearance in each row of the 'findings' column
        total_count = data.str.count(string).sum() # Sum up the count for all rows to get the total number of appearances

        return total_count


    def cosine_similarity(text1, text2): # function that measures distance between tokens
        doc1 = nlp(text1)
        doc2 = nlp(text2)

        return doc1.similarity(doc2) # Compute cosine similarity between two documents

    def test_similarity(text, real_findings): # tret
        similarity = [cosine_similarity(text, real) for real in real_findings]

        return max(similarity) - findings_diversity

    def generate_output(real, num_gen)
        generate_findings(real,num_gen) #outputs a dataframe with num_gen rows with a new column on generated findings
        sample['medical_report?'] = sample.gen_findings.map(single_text) #add a column from classifier
        sample["in_corpus?"] = sample.gen_findings.map(calculate_appearance) #add a column counting number of exact appearance in the original findings
        sample['similarity'] = sample.gen_findings.apply(test_similarity, args = (test_gen.findings,)) # measures max distance - average distance
        ouput = sample['Indications','Extent of Exam', 'gen_findings', 'medical_report?','in_corpus?', 'similarity' ] # subset of compete datafield
        return output
    return output
