import streamlit as st

import numpy as np
import pandas as pd
import json
import requests
import time
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import tokenizers
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
import os


def get_session_state():
    session_state = st.session_state
    if "my_variable" not in session_state:
        session_state.my_variable = 0
    return session_state

session_state = get_session_state()

@st.cache_data
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

    return tokenizer, model

def single_text(custom_sentence):
    filename = '../../model.pkl'
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


def endoCall(search_input,gif):
    API_TOKEN = "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"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/tombrooks248/EndoGPT"
    def query(payload):
        data = json.dumps(payload)
        response = requests.request("POST", API_URL, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))
    data = query(
        {
            "inputs": search_input,
            "options":{"use_cache":False, "wait_for_model":True},
            "parameters": {
                            "do_sample": True,
                            "top_k": 1,
                            "min_length":30,
                            "max_length":150,
                        },
        }
    )
    gif.empty()
    return data

# def progress_bar():
#     progress_text = "Operation in progress. Please wait."
#     my_bar = st.progress(0, text=progress_text)

    # for percent_complete in range(100):
    #     time.sleep(0.3)
    #     my_bar.progress(percent_complete + 1, text=progress_text)


data = None
mr_bean = None
html_string = '<iframe src="https://giphy.com/embed/QBd2kLB5qDmysEXre9" width="480" height="288" frameBorder="0"</iframe>'

from PIL import Image




st.write(os.getcwd())
image = Image.open('/website/app/Logo.png')
#image = Image.open('../images/Logo.png')

st.image(image, width=300)

st.markdown("""# EndoGP-T
    by Open Medicine""")

st.markdown("#### Please generate a medical report")

search_input = ""

with st.form("select_box_form"):
   user_input = st.selectbox(
    'Select an Input',
    ('INDICATIONS FOR PROCEDURE: Abdominal Pain Nausea and/or Vomiting Other- diarrhoea Extent of Exam:  D2',
     'INDICATIONS FOR PROCEDURE: Longstanding history of IDA and high eosinophilis in blood. upper abdo pain /alternatin diarhoea/constipaton Extent of Exam:  D1  FINDINGS: ',
     'INDICATIONS FOR PROCEDURE: Positive coeliac Extent of Exam:  Failed intubation  FINDINGS:'))

   # Every form must have a submit button.
   submitted = st.form_submit_button("Select Submit")
   if submitted:
       search_input = user_input
       #progress_bar()
       mr_bean = st.markdown(html_string, unsafe_allow_html=True)
       data = endoCall(search_input, mr_bean)
       #data = endoCall(search_input)





with st.form("text_form"):
   user_input = st.text_input("write an Input", 'INDICATIONS:      Extent of Exam    FINDINGS:')

   # Every form must have a submit button.
   submitted = st.form_submit_button("Text Submit")
   if submitted:
        search_input = user_input
        #progress_bar()
        mr_bean = st.markdown(html_string, unsafe_allow_html=True)
        data = endoCall(search_input, mr_bean)
        data = endoCall(search_input)


#gen_text='error'
if data:
    gen_text = data[0]["generated_text"]

    html_str = f"""
    <style>
    p.a {{
    font: bold 20px Courier;
    }}
    </style>
    <p class="a">{gen_text}</p>
    """
    session_state.my_variable = gen_text

    st.markdown(html_str, unsafe_allow_html=True)

#gen_text='cat'

st.markdown("#### Is your medical report realistic? 🤔")


button_clicked = st.button("Click me to check!")


# Define the output
#output = st.empty()
# Write the prediction function
def predict(input1):
    # Use the scikit-learn model to generate predictions
    #st.markdown(input1)
    prediction = single_text(input1)
    # Return the results to the output box
    st.markdown("#### The chances of your text being an *endoscopic medical report* are:")

    st.markdown('#### '+ ':green['+str(round(prediction*100,2))+'%]')

    if prediction>0.8:
        st.markdown('#### Your text :green[is probably] a medical report 👍🎈')
    else:
        st.markdown('#### Your text :red[probably is not] a medical report 😔')

if button_clicked:
    st.markdown('Checking the following text:')
    st.markdown(session_state.my_variable)
    hospital_reg = r"\.*FINDINGS:.*"
    line = re.findall(hospital_reg, session_state.my_variable)[0][10:]
    predict(line)
