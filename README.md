
TITANIC SURVIVAL RATE A/B TESTING APP


OVERVIEW
This project is a Streamlit web application that performs a test to determine how long users take when answering the following business question:
    "Which class of passengers had the highest survival rate on the Titanic?"
The app randomly displays one of two different visualizations and records the user's response time to answer the question. The dataset is dynamically retrieved from Google Sheets, ensuring that the app reflects any updates made to the dataset.


FEATURES
- Question is clearly displayed in both the title and after the chart visualization
- Random A/B testing for two different visualizations
- User must answer a multiple-choice question based on the chart
- Time taken to answer is measured and displayed
- Data is loaded dynamically from a Google Sheet
- Data visualization using Seaborn and Matplolib


DATASET
The app uses the Titanic dataset from Seaborn
It includes relevant passenger data such as:
    - Passenger class
    - Survival status

The dataset is cleaned by removing null values before being uploaded to Google Sheets, ensuring compatibility with the Google API


VISUALIZATIONS USED
The app randomly selects one of two charts:
    1. Pie Chart (Survival Rate by Class)
        Shows the percentage of survivors vs non-survivors for each class
    2. Stacked Bar Chart (Survival Count by Class)
        Displays the total number of passengers who survived and died in each class
These charts can both be used to answer the business question.


HOW IT WORKS
1. The user clicks "Show a Chart" --> one of the two visualizations is displayed
2. A multiple-choice question appears asking "Which class had the highest survival rate?"
3. The user selects their answer and clicks "Submit"
4. The app calculates the correct answer based on the dataset, which, unless changed, will be "Second class"
5. The app displays whether the answer was correct and shows the time taken to respond


PROJECT STRUCTURE
DV_Assignment1
|---.streamlit/
|   |---secrets.toml                    # Google Sheets shareable link
|---API_transfer_to_gs.py               # Script to populate the google sheets with the dataset
|---individual_assignment1.py           # Main Streamlit app
|---.gitignore
|---requirements.txt                    # Python dependencies
|---README.md