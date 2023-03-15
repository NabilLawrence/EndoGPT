import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import tokenizers
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

@st.cache_data
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

    return tokenizer, model

def single_text(custom_sentence):
    #filename = '../../model.pkl'
    filename = "website/app/model.pkl"
    loaded_model = pickle.load(open(filename, 'rb'))
    tokenizer, model = load_model()
    #-----------------------------------------------------
    inputs = tokenizer(custom_sentence, return_tensors="pt")
    #-----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
            outputs = model(**inputs)
    #outputs.last_hidden_state.shape
    #-----------------------------------------------------
    preds_prob = loaded_model.predict_proba(np.array(outputs.last_hidden_state[:,0].cpu().numpy()))
    return preds_prob[0][0]

def main():

    st.markdown("#### Is your medical report realistic? 🤔")

    with st.form("my_form"):
        input1 = st.text_area("Enter some text:", value="", height=150)
        submit_button = st.form_submit_button(label="Submit")
    # Define the output
    output = st.empty()
    # Write the prediction function
    def predict(input1):
        # Use the scikit-learn model to generate predictions
        prediction = single_text(input1)
        # Return the results to the output box
        st.markdown("#### The chances of your text being an *endoscopic medical report* are:")

        if prediction>0.8:
            st.markdown('#### '+ ':green['+str(round(prediction*100,2))+'%]')
            st.markdown('#### Your text :green[is probably] a medical report 👍🎈')
            st.balloons()
        else:
            st.markdown('#### '+ ':red['+str(round(prediction*100,2))+'%]')
            st.markdown('#### Your text :red[probably is not] a medical report 😔')

    if submit_button:
        st.empty()
        predict(input1)

if __name__ == '__main__':
    main()
