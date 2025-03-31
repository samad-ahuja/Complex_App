import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
import pickle
import io
import base64
import time

# Set page configuration
st.set_page_config(
    page_title="ML Model Trainer",
    page_icon="🤖",
    layout="wide"
)

# Cache for data loading
@st.cache_data
def load_dataset(dataset_name):
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
    elif dataset_name == "Planets":
        return sns.load_dataset("planets")
    else:
        return pd.DataFrame()

# Initialize session state if not exists
if 'trained_model' not in st.session_state:
    st.session_state['trained_model'] = None
if 'model_metrics' not in st.session_state:
    st.session_state['model_metrics'] = {}
if 'feature_importance' not in st.session_state:
    st.session_state['feature_importance'] = None
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'target_col' not in st.session_state:
    st.session_state['target_col'] = None
if 'feature_cols' not in st.session_state:
    st.session_state['feature_cols'] = []
if 'task_type' not in st.session_state:
    st.session_state['task_type'] = None

# App title and description
st.title("ML Model Trainer App")
st.markdown("""
This application allows you to train machine learning models on various datasets. 
You can select features, configure model parameters, and visualize results.
""")

# Create sidebar for dataset selection and feature configuration
with st.sidebar:
    st.header("Dataset Configuration")
    
    # Dataset selection
    dataset_option = st.selectbox(
        "Select Dataset",
        ["Iris", "Titanic", "Diamonds", "Tips", "Penguins", "Planets", "Upload Your Own"]
    )
    
    # Handle custom dataset upload
    if dataset_option == "Upload Your Own":
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns")
                st.session_state['data'] = data
            except Exception as e:
                st.error(f"Error loading the file: {e}")
        else:
            st.info("Please upload a CSV file")
            st.session_state['data'] = None
    else:
        # Load the selected dataset
        data = load_dataset(dataset_option)
        st.session_state['data'] = data
        st.success(f"Successfully loaded {dataset_option} dataset with {data.shape[0]} rows and {data.shape[1]} columns")
    
    # Add a separator
    st.divider()
    
    # Model configuration section
    st.header("Model Configuration")
    
    if st.session_state['data'] is not None:
        # Task type selection
        task_type = st.radio("Select Task Type", ["Regression", "Classification"])
        st.session_state['task_type'] = task_type
        
        # Target variable selection
        target_col = st.selectbox("Select Target Variable", st.session_state['data'].columns)
        st.session_state['target_col'] = target_col
        
        # Feature selection
        st.subheader("Feature Selection")
        
        # Divide features into numerical and categorical
        numerical_cols = st.session_state['data'].select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = st.session_state['data'].select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target column from features
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        # Initialize empty lists for selected features
        selected_numerical = []
        selected_categorical = []

        # Numerical feature selection with sliders (quantitative)
        if numerical_cols:
            st.subheader("Numerical Features")
            for col in numerical_cols:
                if st.checkbox(f"Use {col}", value=True):
                    selected_numerical.append(col)

        # Categorical feature selection with multiselect (qualitative)
        if categorical_cols:
            st.subheader("Categorical Features")
            for col in categorical_cols:
                if st.checkbox(f"Use {col}", value=True):
                    selected_categorical.append(col)

        # Combine selected features
        selected_features = selected_numerical + selected_categorical
        st.session_state['feature_cols'] = selected_features
        
        if not selected_features:
            st.warning("Please select at least one feature")
        
        # Add a separator
        st.divider()
        
        # Model selection and parameters
        st.header("Model Selection")
        
        if task_type == "Regression":
            model_type = st.selectbox("Select Regression Model", ["Linear Regression", "Random Forest Regressor"])
        else:
            model_type = st.selectbox("Select Classification Model", ["Logistic Regression", "Random Forest Classifier"])
        
        # Model-specific parameters
        st.subheader("Model Parameters")
        
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
        
        if "Random Forest" in model_type:
            n_estimators = st.slider("Number of Estimators", 10, 500, 100)
            max_depth = st.slider("Max Depth", 1, 50, 10)
            min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
        elif "Logistic Regression" in model_type:
            c_value = st.slider("C (Regularization)", 0.01, 10.0, 1.0)
            max_iter = st.slider("Max Iterations", 100, 2000, 1000)
        
        # Training button outside the sidebar to make it more prominent
        st.divider()

# Main area for data preview, training, and visualization
if st.session_state['data'] is not None:
    # Determine dataset title
    if dataset_option == "Upload Your Own":
        dataset_title = st.text_input("Enter a title for your dataset:", "Custom Dataset")
    else:
        dataset_title = f"{dataset_option} Dataset"

    # Display dataset title above preview
    st.markdown(f"## 📊 {dataset_title} Preview")

    # Clean nulls: drop all-null rows, fill numeric with mean, categorical with mode
    df = st.session_state['data'].copy()
    df.dropna(how='all', inplace=True)
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].mean())
        elif df[col].dtype in ['object', 'category']:
            if df[col].nunique() > 0:
                df[col].fillna(df[col].mode()[0])
    df.reset_index(drop=True, inplace=True)
    st.session_state['data'] = df

    st.dataframe(st.session_state['data'].head())
    
    # Data information
    # Enhanced Dataset Information
    st.subheader("Dataset Information")

    df_info = {
        "Column Name": [],
        "Non-Null Count": [],
        "Dtype": []
    }

    for col in st.session_state["data"].columns:
        df_info["Column Name"].append(col)
        df_info["Non-Null Count"].append(st.session_state["data"][col].notnull().sum())
        df_info["Dtype"].append(str(st.session_state["data"][col].dtype))

    df_info_df = pd.DataFrame(df_info)

    # Show as a nice table
    st.dataframe(df_info_df.style.format().set_properties(**{
        'text-align': 'left',
        'border-color': 'lightgray',
        'border-style': 'solid',
        'border-width': '1px'
    }).hide(axis="index"))

    
    # Additional data statistics
    st.subheader("Dataset Statistics")
    st.write(st.session_state['data'].describe())
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Model Training", "Performance Metrics", "Feature Importance"])
    
    with tab1:
        st.header("Model Training")
        
        if st.session_state['feature_cols'] and st.session_state['target_col']:
            # Create a form for model training to improve user experience
            with st.form(key="model_training_form"):
                st.write("Configure all parameters, then press 'Fit Model' to train.")
                
                # Display selected features and target
                st.write(f"**Target Variable:** {st.session_state['target_col']}")
                st.write(f"**Selected Features:** {', '.join(st.session_state['feature_cols'])}")
                
                # Train model button
                submit_button = st.form_submit_button(label="Fit Model")
                
                if submit_button:
                    try:
                        # Get the data
                        X = st.session_state['data'][st.session_state['feature_cols']]
                        y = st.session_state['data'][st.session_state['target_col']]
                        
                        # Check for categorical features and encode them
                        X = pd.get_dummies(X, drop_first=True)
                        
                        # Handle missing values
                        X = X.fillna(X.mean())

                        # Remove rows where y is NaN
                        valid_indices = y.notna()
                        X = X[valid_indices]
                        y = y[valid_indices]

                        # Split the data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )
                        
                        # Show a spinner during training
                        with st.spinner("Training model..."):
                            # Initialize the model based on selection
                            if st.session_state['task_type'] == "Regression":
                                if model_type == "Linear Regression":
                                    model = LinearRegression()
                                elif model_type == "Random Forest Regressor":
                                    model = RandomForestRegressor(
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        random_state=42
                                    )
                            else:  # Classification
                                if model_type == "Logistic Regression":
                                    model = LogisticRegression(
                                        C=c_value,
                                        max_iter=max_iter,
                                        random_state=42
                                    )
                                elif model_type == "Random Forest Classifier":
                                    model = RandomForestClassifier(
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        random_state=42
                                    )
                            
                            # Add artificial delay to simulate complex training
                            time.sleep(1)
                            
                            # Train the model
                            model.fit(X_train, y_train)
                            
                            # Make predictions
                            y_pred = model.predict(X_test)
                            
                            # Store the model and predictions in session state
                            st.session_state['trained_model'] = model
                            st.session_state['predictions'] = y_pred
                            
                            # Calculate metrics
                            if st.session_state['task_type'] == "Regression":
                                mse = mean_squared_error(y_test, y_pred)
                                r2 = r2_score(y_test, y_pred)
                                st.session_state['model_metrics'] = {
                                    'mse': mse,
                                    'r2': r2,
                                    'y_test': y_test,
                                    'y_pred': y_pred
                                }
                            else:  # Classification
                                accuracy = accuracy_score(y_test, y_pred)
                                conf_matrix = confusion_matrix(y_test, y_pred)
                                
                                # For ROC curve (if binary classification)
                                if len(np.unique(y)) == 2:
                                    if hasattr(model, "predict_proba"):
                                        y_proba = model.predict_proba(X_test)[:, 1]
                                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                                        roc_auc = auc(fpr, tpr)
                                    else:
                                        fpr, tpr, roc_auc = None, None, None
                                else:
                                    fpr, tpr, roc_auc = None, None, None
                                
                                st.session_state['model_metrics'] = {
                                    'accuracy': accuracy,
                                    'conf_matrix': conf_matrix,
                                    'class_report': classification_report(y_test, y_pred, output_dict=True),
                                    'fpr': fpr,
                                    'tpr': tpr,
                                    'roc_auc': roc_auc,
                                    'y_test': y_test,
                                    'y_pred': y_pred
                                }
                            
                            # Get feature importance if available
                            if hasattr(model, 'coef_'):
                                if model_type == "Logistic Regression" and len(model.coef_.shape) > 1:
                                    importance = np.mean(np.abs(model.coef_), axis=0)
                                else:
                                    importance = model.coef_
                                feature_names = X.columns
                                st.session_state['feature_importance'] = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': importance
                                }).sort_values(by='Importance', ascending=False)
                            elif hasattr(model, 'feature_importances_'):
                                importance = model.feature_importances_
                                feature_names = X.columns
                                st.session_state['feature_importance'] = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': importance
                                }).sort_values(by='Importance', ascending=False)
                            
                        st.success(f"Model training completed successfully! Please check the Performance Metrics tab for results.")
                    
                    except Exception as e:
                        st.error(f"Error during model training: {e}")
        else:
            st.warning("Please select target variable and at least one feature to train the model.")
    
    with tab2:
        st.header("Performance Metrics")
        
        if st.session_state['trained_model'] is not None and st.session_state['model_metrics']:
            if st.session_state['task_type'] == "Regression":
                st.subheader("Regression Metrics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Mean Squared Error (MSE)", f"{st.session_state['model_metrics']['mse']:.4f}")
                
                with col2:
                    st.metric("R² Score", f"{st.session_state['model_metrics']['r2']:.4f}")
                
                # Residual plot
                st.subheader("Residual Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                residuals = st.session_state['model_metrics']['y_test'] - st.session_state['model_metrics']['y_pred']
                
                sns.histplot(residuals, kde=True, ax=ax)
                ax.set_xlabel("Residual Value")
                ax.set_ylabel("Frequency")
                ax.set_title("Distribution of Residuals")
                
                st.pyplot(fig)
                
                # Actual vs Predicted plot
                st.subheader("Actual vs Predicted Values")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.scatter(
                    st.session_state['model_metrics']['y_test'],
                    st.session_state['model_metrics']['y_pred'],
                    alpha=0.5
                )
                
                # Add perfect prediction line
                min_val = min(st.session_state['model_metrics']['y_test'].min(), st.session_state['model_metrics']['y_pred'].min())
                max_val = max(st.session_state['model_metrics']['y_test'].max(), st.session_state['model_metrics']['y_pred'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Actual vs Predicted Values")
                
                st.pyplot(fig)
                
            else:  # Classification
                st.subheader("Classification Metrics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Accuracy", f"{st.session_state['model_metrics']['accuracy']:.4f}")
                
                with col2:
                    # Extract precision and recall from classification report
                    class_report = st.session_state['model_metrics']['class_report']
                    
                    if 'weighted avg' in class_report:
                        weighted_precision = class_report['weighted avg']['precision']
                        weighted_recall = class_report['weighted avg']['recall']
                        
                        st.metric("Weighted Precision", f"{weighted_precision:.4f}")
                        st.metric("Weighted Recall", f"{weighted_recall:.4f}")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                conf_matrix = st.session_state['model_metrics']['conf_matrix']
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                
                ax.set_xlabel("Predicted Labels")
                ax.set_ylabel("True Labels")
                ax.set_title("Confusion Matrix")
                
                st.pyplot(fig)
                
                # ROC Curve (only for binary classification)
                if st.session_state['model_metrics']['fpr'] is not None:
                    st.subheader("ROC Curve")
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    fpr = st.session_state['model_metrics']['fpr']
                    tpr = st.session_state['model_metrics']['tpr']
                    roc_auc = st.session_state['model_metrics']['roc_auc']
                    
                    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                    ax.legend(loc="lower right")
                    
                    st.pyplot(fig)
                
                # Classification Report
                st.subheader("Classification Report")
                
                # Convert to dataframe for better display
                report_df = pd.DataFrame(st.session_state['model_metrics']['class_report']).transpose()
                
                # Drop support column to focus on precision, recall, and f1-score
                if 'support' in report_df.columns:
                    report_df = report_df.drop('support', axis=1)
                
                st.dataframe(report_df.style.format("{:.4f}"))
                
        else:
            st.info("Please train a model first to see performance metrics.")
    
    with tab3:
        st.header("Feature Importance")
        
        if st.session_state['feature_importance'] is not None:
            # Display feature importance as a table
            st.dataframe(st.session_state['feature_importance'].style.format({"Importance": "{:.4f}"}))
            
            # Plot feature importance
            st.subheader("Feature Importance Visualization")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Sort by importance for better visualization
            sorted_df = st.session_state['feature_importance'].sort_values('Importance')
            
            # Plot horizontal bar chart
            sns.barplot(
                x='Importance',
                y='Feature',
                data=sorted_df.tail(10),  # Show top 10 features
                ax=ax
            )
            
            ax.set_title("Top 10 Feature Importance")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            
            st.pyplot(fig)
        else:
            st.info("Feature importance will be displayed after training a model.")

    # Export model section
    st.header("Export Model")
    
    if st.session_state['trained_model'] is not None:
        # Export the trained model
        model_bytes = pickle.dumps(st.session_state['trained_model'])
        b64 = base64.b64encode(model_bytes).decode()
        href = f'<a href="data:file/pickle;base64,{b64}" download="trained_model.pkl">Download Trained Model</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Export predictions
        if st.session_state['predictions'] is not None:
            pred_df = pd.DataFrame({
                'Actual': st.session_state['model_metrics']['y_test'],
                'Predicted': st.session_state['predictions']
            })
            
            csv = pred_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    else:
        st.info("Train a model first to enable model export.")

else:
    st.info("Please select a dataset to begin.")

# Footer
st.divider()
st.caption("ML Model Trainer App - Created for the Interactive Application Development Assignment")
