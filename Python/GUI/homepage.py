import streamlit as st
import pandas as pd

# st.set_page_config
st.set_page_config(page_title="This is the Page Title", layout="wide")
def sidebar():
    with st.sidebar:
        st.sidebar.header("Header1")
        st.success("TEST")

sidebar()

st.title("This is the Title")
st.header("This is the header")
st.write("Hello World from Streamlit using st.write")

st.markdown(
        """
            AAI1001 Data Engineering and Visualisations Project
            Done by: 
            -   ASHLINDER KAUR DO S. AJAIB SINGH      [2202636]
            -   LEO EN QI VALERIE                     [2202795]
            -   TEO XUANTING                          [2202217]
        """
)