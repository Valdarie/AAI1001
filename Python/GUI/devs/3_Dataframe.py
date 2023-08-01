import streamlit as st
import pandas as pd

# Function to get data from CSV files without caching
def get_data(x):
    return pd.read_csv(x)

# Define search_bar function (unchanged)
def search_bar(df, col_name, ele):
    if col_name in df.columns:
        unique_identifier = df[col_name].unique()
        select = st.selectbox(ele, unique_identifier)
        return df[df[col_name] == select]
    else:
        st.write(f"Column '{col_name}' does not exist in the DataFrame.")
        return df  # Returning the original DataFrame if the column doesn't exist

# Load data from CSV Files
scp_data = get_data('..\Jupyter\scp_statements.csv')
ptbxl_data = get_data('..\Jupyter\ptbxl_database.csv')

# Display SCP Statements
st.header('SCP Statements')

# Text Input Search Bar [DOES NOT WORK YET]
scp_input = st.text_input("Enter the column name for filtering:")
if scp_input:
    selected_df = search_bar(scp_data, scp_input, f"Selected {scp_input}")
    # Display the filtered DataFrame
    st.write("## Filtered SCP Data")
    st.write(selected_df)
else:
    # If no user input, show full data
    st.write(scp_data)  # use st.write to show data

# Display Ptbxl Database
st.header('Ptbxl Database')
ptbxl_input = st.text_input("Enter the column name for filtering: ")
ptbxl_id = st.text_input("Or search by Patient ID: ")

if ptbxl_input:
    selected_df = search_bar(ptbxl_data, ptbxl_input, f"Selected {ptbxl_input}")
    # Display the filtered DataFrame
    st.write("## Filtered ptbxl data")
    st.write(selected_df)

if ptbxl_id:
    ptbxl_id = ptbxl_id.strip()  # Remove any leading / trailing spaces from user input

    # Convert the patient_id column to integers where possible
    ptbxl_data['patient_id'] = pd.to_numeric(ptbxl_data['patient_id'], errors='coerce')

    selected_df = ptbxl_data[ptbxl_data['patient_id'] == int(ptbxl_id)]
    try:
        if not selected_df.empty:
            # Display the filtered DataFrame
            st.write(f"## Patient {ptbxl_id}'s data")
            st.write(selected_df)
        else:
            st.write(f"No data found for Patient ID: {ptbxl_id}")
    except ValueError:
        st.write(f"Please key in an integer")

if not ptbxl_input and not ptbxl_id:
    # If no user input, show full data
    st.write(ptbxl_data)
