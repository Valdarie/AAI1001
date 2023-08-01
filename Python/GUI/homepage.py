import streamlit as st
import os

# st.set_page_config
st.set_page_config(page_title="AAI1001", layout="wide", page_icon="💯")

# Center-align the button in the sidebar
if "show_additional_pages" not in st.session_state:
    st.session_state.show_additional_pages = False

if st.sidebar.button("For Devs"):
    st.session_state.show_additional_pages = not st.session_state.show_additional_pages

st.title("AAI1001 Data Engineering and Visualization Project")
st.header("Cardiovascular Diseases Prediction via Electrocardiogram")

st.markdown(
    """
    Done by:\n
    👧 ASHLINDER KAUR DO S. AJAIB SINGH [2202636]\n
    👧 LEO EN QI VALERIE                [2202795]\n
    👧 TEO XUANTING                     [2202217]
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

# Show the content of Model_Evaluation.py and Dataframe.py when "For Devs" button is clicked
if st.session_state.show_additional_pages:
    st.sidebar.header("Additional Pages")

    # Load and execute the content of Normal_Evaluation.py
    normal_evaluation_path = os.path.join(os.getcwd(), "devs", "2_Normal_Evaluation.py")
    if st.sidebar.button("Normal Evaluation"):
        with open(normal_evaluation_path, "r") as file:
            code = file.read()
        exec(code)