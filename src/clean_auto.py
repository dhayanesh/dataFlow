import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
import seaborn as sb
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve

def clean_auto(df):
    #Replacing unknown (?) values with NaN
    df = df.replace('?', np.nan)

    #Removing duplicate records
    df.drop_duplicates(inplace=True)

    #Dropping columns with high percentage of missing values (> 40%) and IDs
    df = df.drop(['weight', 'payer_code', 'medical_specialty', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id'], axis=1)

    #Dropping additional column with unique identifer which will not be useful predicting readmission
    df = df.drop(['encounter_id', 'patient_nbr'], axis=1) #,'max_glu_serum','A1Cresult'

    df['readmitted'] = df['readmitted'].replace({'NO': 0, '>30': 0, '<30': 1})

    
    def check_diabetes(diag):
        if str(diag).startswith('250'):
            return 1
        else:
            return 0

    #Impute missing values - categorical -> mode values

    race_mode = df['race'].mode()[0]
    df['race'] = df['race'].fillna(race_mode)

    diag1_mode = df['diag_1'].mode()[0]
    df['diag_1'] = df['diag_1'].fillna(diag1_mode)

    diag2_mode = df['diag_2'].mode()[0]
    df['diag_2'] = df['diag_2'].fillna(diag2_mode)

    diag3_mode = df['diag_3'].mode()[0]
    df['diag_3'] = df['diag_3'].fillna(diag3_mode)

    df['has_diabetes'] = df['diag_1'].apply(check_diabetes)

    diag_num = df[['diag_1', 'diag_2', 'diag_3']].notnull().sum(axis=1)
    df['num_diagnoses'] = diag_num

    df = df.drop(["diag_1","diag_2","diag_3"], axis=1)

    #Converting Categorical values to numerical
    col_names = list(df.columns)

    for i in col_names: 
        if str(df[i].dtype) == "object":
            labels = df[i].astype('category').cat.categories.tolist()
            replace_age = {i : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}

            df.replace(replace_age, inplace=True)

    predict_column = df.pop("readmitted")
    df.insert(len(df.columns), "readmitted", predict_column)
    df = df.dropna(axis='columns')
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    st.write(f"<h6 style='text-align:center;'>Cleaned Data (Displaying first 5 rows of the dataset)</h6>", unsafe_allow_html=True)
    st.write(df.head())

    return X, y