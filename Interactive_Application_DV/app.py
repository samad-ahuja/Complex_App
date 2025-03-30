import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc

# Error-handling wrapper
def safe_run():
    try:
        # Set page configuration
        st.set_page_config(
            page_title="ML Model Trainer",
            page_icon="🤖",
            layout="wide"
        )

        # Cache for data loading
        @st.cache_data
        def load_dataset(dataset_name):
            try:
                if dataset_name == "Iris":
                    return sns.load_dataset("iris")
                elif dataset_name == "Titanic":
                    return sns.load_dataset("titanic")
                elif dataset_name == "Diamonds":
                    return sns.load_dataset("diamonds")
                elif dataset_name == "Tips":
                    return sns.load_dataset("tips")
                elif dataset_name == "Penguins":
                    return sns.load_dataset("penguins")
                else:
                    return pd.DataFrame()
            except Exception as e:
                st.error(f"Failed to load dataset '{dataset_name}': {e}")
                return pd.DataFrame()

        # Initialize session state
        for key, default in {
            'trained_model': None,
            'model_metrics': {},
            'feature_importance': None,
            'predictions': None,
            'data': None,
            'target_col': None,
            'feature_cols': [],
            'task_type': None
        }.items():
            if key not in st.session_state:
                st.session_state[key] = default

        # UI Header
        st.title("ML Model Trainer App")
        st.markdown("""
        Train machine learning models on various datasets. 
        Select features, configure model parameters, and visualize results.
        """)

        # Sidebar Configuration
        with st.sidebar:
            st.header("Dataset Configuration")
            dataset_option = st.selectbox("Select Dataset", ["Iris", "Titanic", "Diamonds", "Tips", "Penguins", "Upload Your Own"])

            if dataset_option == "Upload Your Own":
                uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
                if uploaded_file:
                    try:
                        data = pd.read_csv(uploaded_file)
                        st.success(f"Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")
                        st.session_state['data'] = data
                    except Exception as e:
                        st.error(f"Error loading file: {e}")
                        st.session_state['data'] = None
            else:
                data = load_dataset(dataset_option)
                if not data.empty:
                    st.session_state['data'] = data
                    st.success(f"Successfully loaded {dataset_option} dataset")
                else:
                    st.session_state['data'] = None

            st.divider()
            st.header("Model Configuration")

            if st.session_state['data'] is not None and not st.session_state['data'].empty:
                task_type = st.radio("Select Task Type", ["Regression", "Classification"])
                st.session_state['task_type'] = task_type

                target_col = st.selectbox("Select Target Variable", st.session_state['data'].columns)
                st.session_state['target_col'] = target_col

                st.subheader("Feature Selection")

                numerical_cols = st.session_state['data'].select_dtypes(include=['float64', 'int64']).columns.tolist()
                categorical_cols = st.session_state['data'].select_dtypes(include=['object', 'category']).columns.tolist()

                if target_col in numerical_cols:
                    numerical_cols.remove(target_col)
                if target_col in categorical_cols:
                    categorical_cols.remove(target_col)

                selected_numerical = [col for col in numerical_cols if st.checkbox(f"Use {col}", value=True)]
                selected_categorical = [col for col in categorical_cols if st.checkbox(f"Use {col}", value=True)]

                selected_features = selected_numerical + selected_categorical
                st.session_state['feature_cols'] = selected_features

                if not selected_features:
                    st.warning("Please select at least one feature")

                st.divider()
                st.header("Model Selection")

                if task_type == "Regression":
                    model_type = st.selectbox("Select Regression Model", ["Linear Regression", "Random Forest Regressor"])
                else:
                    model_type = st.selectbox("Select Classification Model", ["Logistic Regression", "Random Forest Classifier"])

                st.subheader("Model Parameters")
                test_size = st.slider("Test Size (%)", 10, 50, 20) / 100

                if "Random Forest" in model_type:
                    n_estimators = st.slider("Number of Estimators", 10, 200, 100)
                    max_depth = st.slider("Max Depth", 1, 20, 10)
                elif model_type == "Logistic Regression":
                    c_value = st.slider("C (Regularization)", 0.01, 10.0, 1.0)
                    max_iter = st.slider("Max Iterations", 100, 1000, 500)

        # Main Content
        if st.session_state['data'] is not None and not st.session_state['data'].empty:
            st.header("Dataset Preview")
            st.dataframe(st.session_state['data'].head())
            st.subheader("Dataset Statistics")
            st.write(st.session_state['data'].describe())
        else:
            st.warning("No dataset loaded.")
            return

        st.write("Session State Debug:", dict(st.session_state))

    except Exception as e:
        st.error(f"App failed to start: {e}")

# Run the app
safe_run()
