import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import requests
from datetime import datetime

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import classification_report



#logger

def log(message):
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# Session State Intilisation

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

st.set_page_config("End-to-End SVM", layout="wide")
st.title("End-to-end SVM platform")

# sidebar : Model Settings

st.sidebar.header("SVM settings")
kernel=st.sidebar.selectbox("kernel",["linear","rbf","poly","sigmoid"])
c=st.sidebar.slider("c(Regularization)", 0.01, 10.0, 1.0)
gamma=st.sidebar.selectbox("Gamma",["scale","auto"])

log(f"SVM settings ---> kernel = {kernel} , c={c}, Gamma={gamma}")

# step 1: Data Ingestion

st.header("Step 1: Data Ingestion")
log("step 1 started:  Data Ingestion")

option=st.radio("Choose Data Source ",["Download Dataset","upload CSV"])

df=None
raw_path=None

if option=="Download Dataset":
    if st.button("Download Iris Dataset"):
        log("Downloading Iris Dataset")
        url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response=requests.get(url)

        raw_path=os.path.join(RAW_DIR,"iris.csv")
        with open(raw_path,"wb") as f:
            f.write(response.content)
        
        df=pd.read_csv(raw_path)
        st.success("Dataset Downloaded succesfully")
        log(f"Iris dataset saved at {raw_path}")

if option=="upload CSV":
    uploaded_file=st.file_uploader("upload CSV File", type=["CSV"])
    if uploaded_file:
        raw_path=os.path.join(RAW_DIR,uploaded_file.name)
        with open(raw_path,"wb") as f:
            f.write(uploaded_file.getbuffer())
        df=pd.read_csv(raw_path)
        st.success("File uploaded successfully")
        log(f"Uploaded data saved at {raw_path}")

#------------------------------------------------

#Step 2: EDA

if df is not None:
    st.header("Step 2: Exploratory Data Analysis")
    log("Step 2 Started: EDA")

    st.dataframe(df.head())
    st.write("Shape",df.shape)
    st.write("Missing Values:",df.isnull().sum())

    corr=df.corr(numeric_only=True)
    if not corr.empty:
        fig,ax=plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm",ax=ax)
        st.pyplot(fig)

    log("EDA completed")

#------------------------------------------------
# Step 3 : Data Cleaning

if df is not None:
    st.header("Step 3: Data Cleaning")

    strategy=st.selectbox(
        "Missing value Strategy",["Mean","Median","Drop Rows"]
    )
    
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

#---------------------------------------------------------------------------------------

#Step 4: Save the Cleaned Data

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

#---------------------------------------------------------------

# step 5: load cleaned Dataset

st.header("Step 5 : Load cleaned Dataset")
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

#-------------------------------------------------

from sklearn.model_selection import GridSearchCV

#Step 6: Train SVM
st.header("Step 6: Train SVM")
if "df_model" not in st.session_state:
    st.warning("Please load a cleaned dataset in Step 5 first.")
    st.stop()

df_model = st.session_state.df_model
log("Step 6 started : SVM Training")

target=st.selectbox("Select Target column",df_model.columns)

y=df_model[target]

if y.dtype != "object":
    y = pd.qcut(y, 2, labels=[0, 1])
    log("Continuous target converted to binary classes for SVC")

if y.dtype=="object":
    y=LabelEncoder().fit_transform(y)
    log("Target Column encoded")

x=df_model.drop(columns=[target])
x=x.select_dtypes(include=np.number)

if x.empty:
    st.error("No numeric features available for the training..")
    st.stop()

scaler=StandardScaler()
x=scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)


use_grid = st.checkbox("Use GridSearchCV for hyperparameter tuning")

if use_grid:
    st.info("Performing GridSearchCV. This may take a while...")
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }
    grid = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
    grid.fit(x_train, y_train)
    model = grid.best_estimator_
    st.success(f"Best Parameters: {grid.best_params_}")
    log(f"GridSearchCV completed | Best params: {grid.best_params_}")
else:
    model = SVC(kernel=kernel, C=c, gamma=gamma)
    model.fit(x_train, y_train)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"Model Accuracy: {acc:.2f}")

st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

log("SVM training and evaluation completed")
