#Pip install streamlit 
#Pip installJupytext
import streamlit as st
import jupytext 

st.set_page_config(page_title="AAI1001 | Results", layout="wide", page_icon="📋")

def sidebar():
    with st.sidebar:
        st.header('For Devs:')
        container = st.container()
        with container:
            st.write('Performance Measurement and Evaluation')
sidebar()

st.header("Prediction Results")