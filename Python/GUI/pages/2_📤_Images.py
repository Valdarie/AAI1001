import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from PIL import Image
import numpy as np

st.header("Upload ECG")
container = st.container()

with container:
    uploaded_files = st.file_uploader("Choose an ECG file", accept_multiple_files=True, type=['png', 'dat'])
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)
        st.write(bytes_data)
    st.write('This is inside the container.')

