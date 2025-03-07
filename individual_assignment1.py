import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from streamlit_gsheets import GSheetsConnection
import gspread
from google.oauth2.service_account import Credentials

dataset_name = 'titanic'

#Export the seaborn dataset to Google Sheets
#Importing Google Sheets API
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credential_dict = st.secrets["google_service_account"]
credentials = Credentials.from_service_account_info(credential_dict)
client = gspread.authorize(credentials)

#Linking Empty Google Sheets to the environment
sheet_name = "DV_Assignment1"
sheet = client.open(sheet_name).sheet1

#Clean the seaborn dataset before uploading to Google Sheet --> remove rows with null values
dataset = sns.load_dataset(dataset_name)
dataset = dataset.dropna()

#Upload the dataset to Google Sheets
gs_dataset = [dataset.columns.tolist()] + dataset.values.tolist()
sheet.update('A1', gs_dataset)
print("Data successfully uploaded to Google Sheets!")

#Create a connection object and load data from the Google Sheet
conn = st.connection("gsheets", type = GSheetsConnection)
data = conn.read(worksheet = "sheet", ttl = "10m")

#Streamlit UI
st.title("Which Class of Passengers Survived the Most")
st.write("This app tests which visualization best answers the question above")

#Create the state environment for the charts
if 'chart' not in st.session_state:
    st.session_state.chart = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = None

#Plot Both Visualizations
#Visualization 1: Survival rate
def plot_chart_a():
    fig, ax = plt.subplots(1, len(data['Pclass'].unique()), figsize=(12,4))
    for i, pclass in enumerate(sorted(data['Pclass'].unique())):
        class_data = data[data['Pclass'] == pclass]['Survived'].value_counts()
        ax[i].pie(class_data, labels = ["Did not survive", "Survived"], autopct = '%1.1f%%', colors = ['red', 'green'])
        ax[i].set_title(f"Class {pclass} Survival")
    st.pyploy(fig)

#Visualization 2: Survival count
def plot_chart_b():
    fig, ax = plt.subplots()
    sns.histplot(data=data, x='Pclass', hue='Survived', multiple='stack', discrete=True, shrink=0.8, ax=ax)
    ax.set_title("Survival Count by Passenger Class")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Count")
    st.pyplot(fig)

#Create a button to randomly show a chart after being clicked
if st.button("Show a Chart"):
    st.session_state.chart = random.choice(["A", "B"])
    st.session_state.start_time = time.time()

#Show the chart
if st.session_state.chart:
    if st.session_state.chart == "A":
        plot_chart_a()
    else:
        plot_chart_b()
    
    #Record the time taken to answer the question
    if st.button("I answered your question"):
        elapsed_time = time.time() - st.session_state.start_time
        st.write(f"You took {elapsed_time:.2f} seconds to answer the question.")

#Streamlit deployment and Github repo
st.markdown("### [GitHub Repository](https://github.com/samad-ahuja/DV_Assignment1.git)")
st.markdown("### [Live Streamlit App](https://dv-individual-assignment1.streamlit.app/)")