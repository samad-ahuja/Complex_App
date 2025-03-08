import seaborn as sns
import streamlit as st
from streamlit_gsheets import GSheetsConnection
import gspread
from google.oauth2.service_account import Credentials
import individual_assignment1 as ia1

#Export the seaborn dataset to Google Sheets
#Importing Google Sheets API
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.file"]
credential_dict = st.secrets["google_service_account"]
credentials = Credentials.from_service_account_info(credential_dict, scopes = scope)
client = gspread.authorize(credentials)
print("Using service account email: ", credentials.service_account_email)

#Linking Empty Google Sheets to the environment
sheets = client.openall()
print("Available Sheets:", [s.title for s in sheets])

#Opening sheet
sheet_name = "Streamlit Data"
sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/13qsVw96p3-XAqH0A4_w0Igk8rduEcPyQIER2IC57Sfk/edit?gid=0#gid=0").sheet1
print(f"Successfully accessed sheet: {sheet_name}")

#Clean the seaborn dataset before uploading to Google Sheet --> remove rows with null values
dataset = sns.load_dataset(ia1.dataset_name)
dataset = dataset.dropna()

#Upload the dataset to Google Sheets
gs_dataset = [dataset.columns.tolist()] + dataset.values.tolist()
sheet.update(values = gs_dataset, range_name = 'A1')
print("Data successfully uploaded to Google Sheets!")
