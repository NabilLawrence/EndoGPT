import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import tokenizers
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#-------------------------------------------------------------------------------
st.set_page_config(
    page_title="EndoGP-T",
    page_icon="📖",
)
#-------------------------------------------------------------------------------


def main():

    #st.markdown("#### Is your medical report realistic? 🤔")

    # Open medicine}
    st.markdown("---")

    st.markdown("### Medical data science unchained")

    st.markdown("---")

    st.markdown("### Objectives")

    st.markdown("""
    1. Generate Virtual Medical Reports from real data.
    2. Classify if the generated output is fake or not.
    3. Show proof of concept by generating synthesized Endoscopic reports.
    """)

    st.markdown("---")

    st.markdown("### Related projects")

    st.markdown("""

    - https://github.com/NabilLawrence/EndoGPT

    - https://huggingface.co/tombrooks248/EndoGPT

    - https://docs.ropensci.org/EndoMineR/articles/EndoMineR.html

    - https://github.com/sebastiz/FakeEndoReports
    """)
    st.markdown("---")

    st.markdown("### Contributors")

    st.markdown("""

    - Nabil Lawrence Jamal
    - Tom Brooks
    - Karim Id-Boufker
    - Oscar Rincon
    - Oliver Giles
    - Andrei Danila
    """)

    st.markdown("---")



if __name__ == '__main__':
    main()
