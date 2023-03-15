import streamlit as st

import numpy as np
import pandas as pd
import json
import requests
import time
import os

#-------------------------------------------------------------------------------
st.set_page_config(
    page_title="EndoGP-T",
    page_icon="✏️",
)
#-------------------------------------------------------------------------------

def endoCall(search_input):
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
    return data

def progress_bar():
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.3)
        my_bar.progress(percent_complete + 1, text=progress_text)


data = None

from PIL import Image

#st.write(os.getcwd())
#image = Image.open('../images/Logo.png')

#st.image(image, width=300)

#st.markdown("""# EndoGP-T
#    by Open Medicine""")
st.markdown("#### Please generate a medical report ")

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
       data = endoCall(search_input)





with st.form("text_form"):
   user_input = st.text_input("write an Input", 'INDICATIONS:      Extent of Exam    FINDINGS:')

   # Every form must have a submit button.
   submitted = st.form_submit_button("Text Submit")
   if submitted:
        search_input = user_input
        progress_bar()
        data = endoCall(search_input)



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

    st.markdown(html_str, unsafe_allow_html=True)
    st.balloons()
