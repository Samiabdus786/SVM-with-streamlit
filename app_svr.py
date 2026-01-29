import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import requests
from datetime import datetime

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#logger
def log(message):
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# Session State
if "cleaned_saved" not in st.session_state:
    st.session_state.cleaned_saved=False
if "df_clean" not in st.session_state:
    st.session_state.df_clean=None

# Folder Setup
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
RAW_DIR=os.path.join(BASE_DIR,'data','raw')
CLEAN_DIR=os.path.join(BASE_DIR,'data','cleaned')
os.makedirs(RAW_DIR,exist_ok=True)
os.makedirs(CLEAN_DIR,exist_ok=True)

log("Application started")
log(f"RAW_DIR={RAW_DIR}")
log(f"CLEAN_DIR={CLEAN_DIR}")

# page config
st.set_page_config("End-to-End SVR", layout="wide")
st.title("End-to-end SVR platform")

# sidebar : Model Settings
st.sidebar.header("SVR settings")
kernel=st.sidebar.selectbox("kernel",["linear","rbf","poly","sigmoid"])
c=st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
epsilon=st.sidebar.slider("epsilon", 0.01, 1.0, 0.1)
gamma=st.sidebar.selectbox("Gamma",["scale","auto"])

log(f"SVR settings ---> kernel = {kernel} , C={c}, Gamma={gamma}, epsilon={epsilon}")

# Step 1: Data Ingestion
st.header("Step 1: Data Ingestion")
log("Step 1 started: Data Ingestion")

option=st.radio("Choose Data Source ",["Download Dataset","upload CSV"])
df=None
raw_path=None

if option=="Download Dataset":
    if st.button("Download Example Dataset"):
        url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response=requests.get(url)
        raw_path=os.path.join(RAW_DIR,"iris.csv")
        with open(raw_path,"wb") as f:
            f.write(response.content)
        df=pd.read_csv(raw_path)
        st.success("Dataset Downloaded succesfully")
        log(f"Dataset saved at {raw_path}")

if option=="upload CSV":
    uploaded_file=st.file_uploader("Upload CSV File", type=["CSV"])
    if uploaded_file:
        raw_path=os.path.join(RAW_DIR,uploaded_file.name)
        with open(raw_path,"wb") as f:
            f.write(uploaded_file.getbuffer())
        df=pd.read_csv(raw_path)
        st.success("File uploaded successfully")
        log(f"Uploaded data saved at {raw_path}")

# Step 2: EDA
if df is not None:
    st.header("Step 2: Exploratory Data Analysis")
    log("Step 2 Started: EDA")
    st.dataframe(df.head())
    st.write("Shape",df.shape)
    st.write("Missing Values:",df.isnull().sum())
    corr=df.corr(numeric_only=True)
    if not corr.empty:
        fig,ax=plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    log("EDA completed")

# Step 3: Data Cleaning
if df is not None:
    st.header("Step 3: Data Cleaning")
    strategy=st.selectbox("Missing value Strategy",["Mean","Median","Drop Rows"])
    df_clean=df.copy()
    if strategy=="Drop Rows":
        df_clean=df_clean.dropna()
    else:
        for col in df_clean.select_dtypes(include=np.number):
            if strategy=="Mean":
                df_clean[col]=df_clean[col].fillna(df_clean[col].mean())
            else:
                df_clean[col]=df_clean[col].fillna(df_clean[col].median())
    st.session_state.df_clean=df_clean
    st.success("Data cleaning completed")
else:
    st.info("Please complete Step 1( Data Ingestion) first...")

# Step 4: Save cleaned data
if st.button("Save cleaned Dataset"):
    if st.session_state.df_clean is None:
        st.error("No cleaned data found. Please Complete step 3 first...")
    else:
        timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        clean_filename=f"cleaned_dataset_{timestamp}.csv"
        clean_path=os.path.join(CLEAN_DIR,clean_filename)
        st.session_state.df_clean.to_csv(clean_path,index=False)
        st.success("Cleaned Data Saved")
        st.info(f"Saved at: {clean_path}")
        log(f"Cleaned dataset saved at {clean_path}")

# Step 5: Load cleaned dataset
st.header("Step 5: Load cleaned Dataset")
clean_files=os.listdir(CLEAN_DIR)
if not clean_files:
    st.warning("No cleaned datasets found. please save one in step 4...")
    log("No cleaned datasets available")
else:
    selected=st.selectbox("Select Cleaned Dataset",clean_files)
    df_model=pd.read_csv(os.path.join(CLEAN_DIR,selected))
    st.session_state.df_model = df_model
    st.success(f"Loaded dataset: {selected}")
    log(f"Loaded cleaned dataset: {selected}")
    st.dataframe(df_model.head())

# Step 6: Train SVR
st.header("Step 6: Train SVR")
if "df_model" not in st.session_state:
    st.warning("Please load a cleaned dataset in Step 5 first.")
    st.stop()

df_model = st.session_state.df_model
log("Step 6 started : SVR Training")

target=st.selectbox("Select Target column",df_model.columns)
y=df_model[target]

x=df_model.drop(columns=[target])
x=x.select_dtypes(include=np.number)

if x.empty:
    st.error("No numeric features available for the training..")
    st.stop()

scaler=StandardScaler()
x=scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

# GridSearchCV toggle
use_grid = st.checkbox("Use GridSearchCV for hyperparameter tuning")

if use_grid:
    st.info("Performing GridSearchCV. This may take a while...")
    param_grid = {
        'C':[0.1,1,10],
        'kernel':['linear','rbf','poly','sigmoid'],
        'gamma':['scale','auto'],
        'epsilon':[0.01,0.1,0.2]
    }
    grid = GridSearchCV(SVR(), param_grid, cv=5, n_jobs=-1)
    grid.fit(x_train, y_train)
    model = grid.best_estimator_
    st.success(f"Best Parameters: {grid.best_params_}")
    log(f"GridSearchCV completed | Best params: {grid.best_params_}")
else:
    model = SVR(kernel=kernel, C=c, gamma=gamma, epsilon=epsilon)
    model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# Regression metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.success(f"Model Metrics - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

# Plot predictions vs actual
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Actual vs Predicted")
st.pyplot(fig)

log("SVR training and evaluation completed")
