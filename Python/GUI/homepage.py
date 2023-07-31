import streamlit as st
import pandas as pd

# st.set_page_config
st.set_page_config(page_title="AAI1001", layout="wide", page_icon="💯")

# Center-align the button in the sidebar
if "show_additional_pages" not in st.session_state:
    st.session_state.show_additional_pages = False

if st.sidebar.button("For Devs"):
    st.session_state.show_additional_pages = not st.session_state.show_additional_pages

st.title("AAI1001")
st.header("Data Engineering and Visualisations Project")
st.write("Hello World from Streamlit using st.write")

with st.container():
    st.markdown(
        """
        Done by: 
        - ASHLINDER KAUR DO S. AJAIB SINGH      [2202636]
        - LEO EN QI VALERIE                     [2202795]
        - TEO XUANTING                          [2202217]
        """
    )

    st.markdown(
        """
        This project aims to design a minimal viable product (MVP) 
        of a trained Machine Learning (ML) model with a Graphical User Interface (GUI) 
        to predict heart disease. It addresses the need for accurate detection, 
        objective assessments, and efficient usage of resources in diagnosing heart disease.
        """
    )
