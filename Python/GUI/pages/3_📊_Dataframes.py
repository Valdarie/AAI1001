import streamlit as st
import pandas as pd

# st.set_page_config
st.set_page_config(page_title="AAI1001 | Dataframes", layout="wide", page_icon="📊")

st.title("This is the Title")
st.header("This is the header")
st.write("Hello World from Streamlit using st.write")

# Function to get data from CSV files and cache it
@st.cache_data
def get_data(x):
    return pd.read_csv(x)

# Define search_bar function
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
if not scp_input: #If no user input, show full data
    st.write(scp_data) # use st.write to show data

if scp_input:
    selected_df = search_bar(scp_data, scp_input, f"Selected {scp_input}")
    # Display the filtered DataFrame
    st.write("## Filtered SCP Data")
    st.write(selected_df)


# Display Ptbxl Database
st.header('Ptbxl Database')
ptbxl_input = st.text_input("Enter the column name for filtering: ")
ptbxl_id = st.text_input("Or search by Patient ID: ")

if not ptbxl_input and not ptbxl_id: #If no user input, show full data
    st.write(ptbxl_data)

if ptbxl_input:
    selected_df = search_bar(ptbxl_data, ptbxl_input, f"Selected {ptbxl_input}")
    # Display the filtered DataFrame
    st.write("## Filtered ptbxl data")
    st.write(selected_df)

if ptbxl_id and not ptbxl_input:
    ptbxl_id = ptbxl_id.strip() # Remove any leading / trailing spaces from user input
    
    # Display the unique patient IDs before removing commas
    #st.write("Unique Patient IDs (Before Removing Commas):")
    #st.write(ptbxl_data['patient_id'].unique())

    # Convert the patient_id column to strings and remove commas
    #ptbxl_data['patient_id'] = ptbxl_data['patient_id'].astype(str).str.replace(',', '').str.strip()

    # Convert the patient_id column to integers where possible
    ptbxl_data['patient_id'] = pd.to_numeric(ptbxl_data['patient_id'], errors='coerce')

    # Display the unique patient IDs after removing commas and converting to integers
    #st.write("Unique Patient IDs (After Removing Commas and Converting to Integers):")
    #st.write(ptbxl_data['patient_id'].unique())

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