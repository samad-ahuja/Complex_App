import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from streamlit_gsheets import GSheetsConnection
import gspread
from oauth2client.service_account import ServiceAccountCredentials

dataset_name = 'titanic'

#Export the seaborn dataset to Google Sheets
#Importing Google Sheets API
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name(r"C:\Users\samad\Downloads\DataVisualization\credentials.json.json", scope)
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


