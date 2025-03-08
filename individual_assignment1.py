import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
from streamlit_gsheets import GSheetsConnection
from google.oauth2.service_account import Credentials

#Create a connection object and load data from the Google Sheet
conn = st.connection("gsheets", type=GSheetsConnection)
data = conn.read()

#Function to find the survival rate of each passenger class
def find_survival_rate(data: pd.DataFrame, pass_class: str) -> None:
    survival_counts = data.groupby("class")["survived"].value_counts().unstack(fill_value=0)
    survived = survival_counts.loc[pass_class, 1] if 1 in survival_counts.columns else 0
    total = survived + survival_counts.loc[pass_class, 0]
    survival_rate = survived / total
    survival_rate_dict[pass_class] = (survival_rate, 1 - survival_rate)
    return

dataset_name = 'titanic'

#Create the dictionary that will be used for the correct answer in the streamlit app
survival_rate_dict: dict = {}

#Give the user the options to answer the question
options = [
    "First",
    "Second",
    "Third"
    ]

for pclass in options:
    find_survival_rate(data, pclass)

#Streamlit UI
st.title("Which Class of Passengers Survived the Most")
st.header("This app tests which visualization best answers the question above")

#Create the state environment for the charts
if 'chart' not in st.session_state:
    st.session_state.chart = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'answered' not in st.session_state:
    st.session_state.answered = None

#Ask the question
st.write("Answer the following question:")
st.write("Which Class of Passengers had the Highest Survival Rate")

#Plot Both Visualizations
#Visualization 1: Survival rate
def plot_chart_a():
    fig, ax = plt.subplots(1, len(survival_rate_dict), figsize=(12,4))
    for i, stats in enumerate(survival_rate_dict.values()):
        pass_class = next(key for key, value in survival_rate_dict.items() if value == stats)
        ax[i].pie(stats, labels = ["Survived", "Did not Survive"], autopct = '%1.1f%%', colors = ['green', 'red'])
        ax[i].set_title(f"{pass_class}Class Survival")
    st.pyplot(fig)

#Visualization 2: Survival count
def plot_chart_b():
    fig, ax = plt.subplots(figsize = (12, 4))
    sns.histplot(data=data, x='class', hue='survived', multiple='stack', discrete=True, shrink=0.8, ax=ax)
    ax.set_title("Survival Count by Passenger Class")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Count")
    st.pyplot(fig)

#Create a button to randomly show a chart after being clicked
if st.button("Show a Chart"):
    st.session_state.chart = random.choice(["A", "B"])
    st.session_state.start_time = time.time()
    st.session_state.answered = False

#Show the chart
if st.session_state.chart:
    if st.session_state.chart == "A":
        plot_chart_a()
    else:
        plot_chart_b()
    
    #Allow user to answer the question
    user_answer = st.radio("Which class of passengers had the highest survival rate?", options)

    #Record the time taken to answer the question
    if st.button("Submit"):
        elapsed_time = time.time() - st.session_state.start_time
        st.session_state.answered = True

        correct_answer = max(survival_rate_dict.keys(), key=lambda k: survival_rate_dict[k][0])

        if user_answer == correct_answer:
            st.success(f"Correct! You answered in {elapsed_time:.2f} seconds.")
        else:
            st.error(f"Incorrect. The correct answer is: {correct_answer}. You answered in {elapsed_time:.2f} seconds.")

#Streamlit deployment and Github repo
st.markdown("### [GitHub Repository](https://github.com/samad-ahuja/DV_Assignment1.git)")
st.markdown("### [Live Streamlit App](https://titanic-dataset-dv.streamlit.app/)")