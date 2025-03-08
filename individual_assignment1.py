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

'''#Published the Google Sheets after transferring the dataset to the sheet, and processed it as a CSV
data_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRlLcx9qhGGOxRRUqFGskjogGGU4_pPSdYblo7TtmXVncD0HOnWTnf0aBIsaqbIzeuXUqGK2PqIUE11/pub?output=csv"
data = pd.read_csv(data_url)'''

#Create a connection object and load data from the Google Sheet
conn = st.connection("gsheets", type=GSheetsConnection)
data = conn.read()

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
    fig, ax = plt.subplots(1, len(data['class'].unique()), figsize=(12,4))
    for i, pclass in enumerate(sorted(data['class'].unique())):
        class_data = data[data['class'] == pclass]['survived'].value_counts()
        ax[i].pie(class_data, labels = ["Did not survive", "Survived"], autopct = '%1.1f%%', colors = ['red', 'green'])
        ax[i].set_title(f"Class {pclass} Survival")
    st.pyploy(fig)

#Visualization 2: Survival count
def plot_chart_b():
    fig, ax = plt.subplots()
    sns.histplot(data=data, x='class', hue='survived', multiple='stack', discrete=True, shrink=0.8, ax=ax)
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