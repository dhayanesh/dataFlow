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

#import models:
from clean_auto import clean_auto
from logisticRegression import logisticRegression
from decisionTree import decisionTree
from randomForestClassifier import randomForestClassifier
from adaBoost import adaBoost
from XGBoost import XGBoost
from naiveBayes import naiveBayes
import joblib

st.set_option('deprecation.showPyplotGlobalUse', False)



def call_cust(custom_list, file):
    loaded_model = joblib.load(file)
    #custom_list1 = [int(float((i))) for i in custom_list]
    custom_list1 = np.array(custom_list)
    test1 = custom_list1.reshape(1,-1)
    result = loaded_model.predict(test1)
    result1 = loaded_model.predict_proba(test1)
    
    
    if int(result) == 0:
        output = "Readmission not required within 30 days!"
    else:
        output = "Readamission required within 30 days!"
    return output, result1.astype(float)

def basicEDA(df):
    import os
    df = df.replace('?', np.nan)
    df.drop_duplicates(inplace=True)
    count = df['race'].value_counts()

    st.markdown("<h6 style='text-align: center;color: #5fb4fb;'>Here's some EDA of your dataset!</h6>", unsafe_allow_html=True)

    plt.bar(count.index, count.values)
    plt.title('Distribution of Race in Patients')
    plt.xlabel('Race')
    plt.ylabel('Count of Patients')

    #Save the plot as an image
    plt.savefig("race_distribution.png")

    plt.figure(figsize=(8, 6))
    sb.countplot(x='race', hue='readmitted', data=df)
    plt.title('Readmission Count by Race')
    plt.xlabel('Race')
    plt.ylabel('Count')

    #Save the plot as an image
    plt.savefig("race_readmit_distribution.png")

    col1, col2 = st.columns([1, 1])
    #First Image
    with col1:
        st.image("race_distribution.png", use_column_width=True, caption="Distribution of Race in Patients")

    #Second Image
    with col2:
        st.image("race_readmit_distribution.png", use_column_width=True, caption="Readmission Count by Race")

    os.remove("race_distribution.png")
    os.remove("race_readmit_distribution.png")



    count = df['time_in_hospital'].value_counts()
    plt.bar(count.index, count.values)
    plt.title('Distribution of time in hospital')
    plt.xlabel('Time in hospital (days)')
    plt.ylabel('Count of Patients')

    #Save the plot as an image
    plt.savefig("time_hospital_distribution.png")

    plt.figure(figsize=(8, 6))
    sb.countplot(x='age', hue='readmitted', data=df)
    plt.title('Readmission Count by Age')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()

    #Save the plot as an image
    plt.savefig("age_readmit_distribution.png")

    col1, col2 = st.columns([1, 1])
    #First Image
    with col1:
        st.image("time_hospital_distribution.png", use_column_width=True, caption="Distribution of time in hospital")

    #Second Image
    with col2:
        st.image("age_readmit_distribution.png", use_column_width=True, caption="Readmission Count by Age")

    os.remove("time_hospital_distribution.png")
    os.remove("age_readmit_distribution.png")


    #distribution of readmission
    sb.countplot(x='readmitted', data=df)
    plt.title('Distribution of Readmission')
    plt.xlabel('Readmission class')
    plt.ylabel('Count')
    
    #Save the plot as an image
    plt.savefig("readmit_distribution.png")

    #distribution of patients by age and gender
    plt.figure(figsize=(12,6))
    sb.countplot(x='age', hue='gender', data=df)

    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Gender and Age Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.2)

    #Save the plot as an image
    plt.savefig("gender_age_distribution.png")
    
    col1, col2 = st.columns([1, 1])
    #First Image
    with col1:
        st.image("readmit_distribution.png", use_column_width=True, caption="Distribution of time in hospital")

    #Second Image
    with col2:
        st.image("gender_age_distribution.png", use_column_width=True, caption="Readmission Count by Age")

    os.remove("readmit_distribution.png")
    os.remove("gender_age_distribution.png")    

    st.markdown('<hr style="border-top: 1px solid; width: 75%; margin: 0 auto;">', unsafe_allow_html=True)

    eda_option = st.selectbox("Do you want to peform more EDA on the dataset?", ["--Select--", "Yes", "No"])

    if eda_option == "Yes":
        sb.boxplot(x=df["num_medications"], y=df['age'])
        plt.title('Box Plot of Number of Medications by Age')
        plt.xlabel('Number of Medications')
        plt.ylabel('Age')

        #Save the plot as an image
        plt.savefig("num_medications_boxplot.png")

        sb.boxplot(x=df["num_lab_procedures"], y=df['age'])
        plt.title('Box Plot of Number of Lab Procedures by Age')
        plt.xlabel('Number of Lab Procedures')
        plt.ylabel('Age')

        # Save the plot as an image
        plt.savefig("num_lab_boxplot.png")


        col1, col2 = st.columns([1, 1])
        #First Image
        with col1:
            st.image("num_medications_boxplot.png", use_column_width=True, caption="Box Plot of Number of Medications by Age")

        #Second Image
        with col2:
            st.image("num_lab_boxplot.png", use_column_width=True, caption="Box Plot of Number of Lab Procedures by Age")

        os.remove("num_medications_boxplot.png")
        os.remove("num_lab_boxplot.png")    




        correlation_matrix = df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))

        #Plotting correlation matrix as a heatmap
        sb.heatmap(correlation_matrix, annot=True, ax=ax)

        plt.title('Correlation Matrix')
        plt.xlabel('Features')
        plt.ylabel('Features')

        #Save the plot as an image
        plt.savefig("correlation_matrix.png")


        num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
        corr_matrix = df[num_cols].corr()

        #Create a figure and axes
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the correlation matrix heatmap
        sb.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)

        # Set the title and labels
        plt.title('Correlation Analysis')
        plt.xlabel('Features')
        plt.ylabel('Features')

        #Save the plot as an image
        plt.savefig('correlation_heatmap.png')

        col1, col2 = st.columns([1, 1])
        #First Image
        with col1:
            st.image("correlation_heatmap.png", use_column_width=True, caption="Correlation Matrix Heatmap of numerical features")

        #Second Image
        with col2:
            st.image("correlation_matrix.png", use_column_width=True, caption="Correlation Matrix")

        os.remove("correlation_matrix.png")
        os.remove("correlation_heatmap.png")   

       # import plotly.express as px

        #fig = px.histogram(df, x='race', color='readmitted')

        #Set the layout
       # fig.update_layout(
       #     title='Readmission Count by Race',
       #     xaxis_title='Race',
        #    yaxis_title='Count'
       # )

        #Display the plot using Streamlit
        #st.plotly_chart(fig, use_container_width=True)



        st.markdown('<hr style="border-top: 1px solid; width: 75%; margin: 0 auto;">', unsafe_allow_html=True) 


        race_readmit_probs = df.groupby('race')['readmitted'].value_counts(normalize=True).mul(100)
        st.markdown("<h6 style='text-align: center;'>Probability of readmission by race</h6>", unsafe_allow_html=True)
        st.text(race_readmit_probs)

        st.markdown('<hr style="border-top: 1px solid; width: 75%; margin: 0 auto;">', unsafe_allow_html=True)
    
    interactive_option = st.selectbox("Do you want to enter into interactive EDA mode?", ["--Select--", "Yes", "No"])
    import plotly.graph_objs as go
    import plotly.express as px
    if interactive_option == "Yes":

        #Distribution of Race in Patients
        count = df['race'].value_counts()
        fig1 = px.bar(df, x=count.index, y=count.values, labels={'x': 'Race', 'y': 'Count of Patients'})
        fig1.update_layout(title='Distribution of Race in Patients')

        #Readmission Count by Race
        fig2 = px.histogram(df, x='race', color='readmitted')
        fig2.update_layout(title='Readmission Count by Race', xaxis_title='Race', yaxis_title='Count')

        #Display the plots
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

        #Distribution of time in hospital
        count = df['time_in_hospital'].value_counts()
        fig3 = px.bar(df, x=count.index, y=count.values, labels={'x': 'Time in hospital (days)', 'y': 'Count of Patients'})
        fig3.update_layout(title='Distribution of time in hospital')

        #Readmission Count by Age
        fig4 = px.histogram(df, x='age', color='readmitted')
        fig4.update_layout(title='Readmission Count by Age', xaxis_title='Age', yaxis_title='Count')

        #Display the plots
        st.plotly_chart(fig3, use_container_width=True)
        st.plotly_chart(fig4, use_container_width=True)

        #Distribution of Readmission
        fig5 = px.histogram(df, x='readmitted')
        fig5.update_layout(title='Distribution of Readmission', xaxis_title='Readmission class', yaxis_title='Count')

        #Gender and Age Distribution
        fig6 = px.histogram(df, x='age', color='gender', barmode='group')
        fig6.update_layout(title='Gender and Age Distribution', xaxis_title='Age', yaxis_title='Count')

        #Display plots
        st.plotly_chart(fig5, use_container_width=True)
        st.plotly_chart(fig6, use_container_width=True)

        int_eda_option = st.selectbox("Do you want to view more interactive EDA?", ["--Select--", "Yes", "No"])

        if int_eda_option == "Yes":
            fig1 = px.box(df, x='num_medications', y='age', labels={'x': 'Number of Medications', 'y': 'Age'})
            fig1.update_layout(title='Box Plot of Number of Medications by Age')

            #Box Plot of Number of Lab Procedures by Age
            fig2 = px.box(df, x='num_lab_procedures', y='age', labels={'x': 'Number of Lab Procedures', 'y': 'Age'})
            fig2.update_layout(title='Box Plot of Number of Lab Procedures by Age')

            #Display plots
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)





st.markdown("<h1 style='text-align: center;color: #5fb4fb;'>DIABETES READMISSION PREDICTOR</h1>", unsafe_allow_html=True)
data_option = st.selectbox("Select an option to make prediction", ["--Select--", "Upload Dataset", "Enter Data"])

if data_option == "Upload Dataset":
    st.markdown("<h6 style='text-align: center;'>Upload your dataset here!</h6>", unsafe_allow_html=True)

    dataset = st.file_uploader("Choose your dataset - CSV file", type="csv")

    X = None
    y = None

    if dataset is not None:
        try:
            data = pd.read_csv(dataset)

            required_columns = ["encounter_id", "patient_nbr", "race", "gender", "age", "weight", "admission_type_id", 
                    "discharge_disposition_id", "admission_source_id", "time_in_hospital", "payer_code", 
                    "medical_specialty", "num_lab_procedures", "num_procedures", "num_medications", 
                    "number_outpatient", "number_emergency", "number_inpatient", "diag_1", "diag_2", "diag_3", 
                    "number_diagnoses", "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide", 
                    "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", 
                    "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", 
                    "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", 
                    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone", "change", 
                    "diabetesMed", "readmitted"]

            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                st.write(f"<p style='color:red'><strong>The following required columns are missing:</strong></p>", unsafe_allow_html=True)
                st.write(f"{', '.join(missing_columns)}")
            else:
                st.write(f"<p style='color:#0FF900'><strong>All required columns are present!</strong></p>", unsafe_allow_html=True)

                st.markdown("<h6 style='text-align: center;'>Displaying first 5 rows of the dataset</h6>", unsafe_allow_html=True)

                st.write(data.head())
            
                basicEDA(data)

                tooltip_text = {
                    'Clean Automatically': 'This is the first option.',
                    'Clean Manually': 'This is the second option.'
                }

                clean_option = st.radio("Select data cleaning option", ["Clean Automatically", "Already Cleaned"], help='Clean Automatically: Select this option to automatically clean the data using predefined cleaning rules. \n\nAlready Cleaned: Select this option if the data is already cleaned and ready for modeling.')

                #tooltip = tooltip_text.get(clean_option, '')
                #st.markdown(f'<span title="{tooltip}">{clean_option}</span>', unsafe_allow_html=True)
                if st.button("Clean!"):
                    if clean_option == "Clean Automatically":
                        st.session_state.X, st.session_state.y = clean_auto(data)
                        #logisticRegression(X, y)
                        st.write(f"<p style='color:#0FF900'><strong>Automatic cleaning successful!</strong></p>", unsafe_allow_html=True)
                    if clean_option == "Already Cleaned":
                        st.write(f"<p style='color:#0FF900'><strong>Please proceed to training if data is already cleaned!</strong></p>", unsafe_allow_html=True)

                model_option = st.radio("Select the model to train your dataset with:", ("Logistic Regression", "Decision Tree", "Random Forest Classifier", "Ada Boost", "XGBoost", "Naive Bayes"), help = 'Logistic Regression: Linear model for classification and regression. \n\n Decision Tree: Tree-based model that makes decisions based on feature values. \n\n Random Forest Classifier: Ensemble of decision trees for classification. \n\n Ada Boost: Ensemble model that combines weak learners to create a strong learner. \n\n XGBoost: Optimized gradient boosting framework for improved model performance. \n\n Naive Bayes: Probabilistic model based on Bayes theorem for classification.')

                if st.button("Train Model!"):
                    #with st.spinner(f"Training with {model_option}..."):
                    #st.write("Clicked Train Model!")
                    #if st.session_state.X is None:
                        #st.write(f"<p style='color:red'><strong>Please clean dataset before training!</strong></p>", unsafe_allow_html=True)
                    #else:
                    if model_option == "Logistic Regression":
                        
                        st.write(f"<p style='color:#0FF900'><strong>Training with Logistic Regression!</strong></p>", unsafe_allow_html=True)
                        #X, y = clean_auto(data)
                        logisticRegression(st.session_state.X, st.session_state.y)

                    elif model_option == "Decision Tree":

                        st.write(f"<p style='color:#0FF900'><strong>Training with Decision Tree!</strong></p>", unsafe_allow_html=True)
                        #X, y = clean_auto(data)
                        decisionTree(st.session_state.X, st.session_state.y)
                    
                    elif model_option == "Random Forest Classifier":
                        
                        st.write(f"<p style='color:#0FF900'><strong>Training with Random Forest Classifier!</strong></p>", unsafe_allow_html=True)
                    # X, y = clean_auto(data)
                        randomForestClassifier(st.session_state.X, st.session_state.y)

                    elif model_option == "Ada Boost":
                        
                        st.write(f"<p style='color:#0FF900'><strong>Training with AdaBoost!</strong></p>", unsafe_allow_html=True)
                        #X, y = clean_auto(data)
                        adaBoost(st.session_state.X, st.session_state.y)

                    elif model_option == "XGBoost":

                        st.write(f"<p style='color:#0FF900'><strong>Training with XGBoost!</strong></p>", unsafe_allow_html=True)
                        #X, y = clean_auto(data)
                        XGBoost(st.session_state.X, st.session_state.y)

                    
                    elif model_option == "Naive Bayes":
                        
                        st.write(f"<p style='color:#0FF900'><strong>Training with Naive Bayes!</strong></p>", unsafe_allow_html=True)
                        #X, y = clean_auto(data)
                        naiveBayes(st.session_state.X, st.session_state.y)

        except Exception as e:
            st.write(f"Error in reading the CSV file: {e}")

elif data_option == "Enter Data":
    race = ""
    gender = ""
    age = ""
    admission_type = ""
    discharge_disposition_id = ""
    admission_source_id = ""
    time_in_hospital = ""
    num_lab_procedures = ""
    num_procedures = ""
    num_medications = ""
    number_outpatient = ""
    number_emergency = ""
    number_inpatient = ""
    diag_1 = ""
    diag_2 = ""
    diag_3 = ""
    number_diagnoses = ""
    metformin = ""
    repaglinide = ""
    nateglinide = ""
    chlorpropamide = ""
    glimepiride = ""
    acetohexamide = ""
    glipizide = ""
    glyburide = ""
    tolbutamide = ""
    pioglitazone = ""
    rosiglitazone = ""
    acarbose = ""
    miglitol = ""
    troglitazone = ""
    tolazamide = ""
    examide = ""
    citoglipton = ""
    glyburide_metformin = ""
    glipizide_metformin = ""
    glimepiride_pioglitazone = ""
    metformin_rosiglitazone = ""
    metformin_pioglitazone = ""
    change = ""
    diabetesMed = ""
    max_glu_serum = ""
    insulin = ""
    A1Cresult = ""
    has_diabetes = ""
    num_diagnoses = ""
    

    st.session_state.race = st.number_input("Enter your race", min_value=0, max_value=100, value=3, step=1, help = "AfricanAmerican: Enter - 1 \n\n Asian: Enter - 2 \n\n Caucasian: Enter - 3 \n\n Hispanic: Enter - 4 \n\n Other: Enter - 5")
    st.session_state.gender = st.number_input("Enter your gender", min_value=0, max_value=100, value=2, step=1, help = "Female: Enter - 1 \n\n Male: Enter - 2 \n\n Others: Enter - 3")
    st.session_state.age = st.number_input("Enter your age", min_value=0, max_value=100, value=9, step=1, help = "Age range and their relevant value to enter: \n\n [0-10): 1 \n\n [10-20): 2 \n\n [20-30): 3 \n\n [30-40): 4 \n\n [40-50): 5 \n\n [50-60): 6 \n\n [60-70): 7 \n\n [70-80): 8 \n\n [80-90): 9 \n\n [90-100): 10")
    #st.session_state.admission_type = st.number_input("Enter your admission type", min_value=0, max_value=100, value=0, step=1, help = "Values from 1 to 8")
    #ad id
    #st.session_state.discharge_disposition_id = st.number_input("Enter your discharge disposition id", min_value=0, max_value=100, value=0, step=1, help = "Values from 1 to 28")
    #st.session_state.admission_source_id = st.number_input("Enter your admission source id", min_value=0, max_value=100, value=0, step=1, help = "Values from 1 to 25")
    st.session_state.time_in_hospital = st.number_input("Enter your time in hospital", min_value=0, max_value=100, value=11, step=1, help = "Values from 1 to 14")
    st.session_state.num_lab_procedures = st.number_input("Enter your number of lab procedures", min_value=0, max_value=100, value=57, step=1, help = "Values from 1 to 132")
    st.session_state.num_procedures = st.number_input("Enter your number of procedures", min_value=0, max_value=100, value=5, step=1, help = "Values from 0 to 6")
    st.session_state.num_medications = st.number_input("Enter your number of medications", min_value=0, max_value=100, value=22, step=1, help = "Values from 1 to 81")
    st.session_state.number_outpatient = st.number_input("Enter your number of out patient", min_value=0, max_value=100, value=0, step=1, help = "Values from 0 to 42")
    st.session_state.number_emergency = st.number_input("Enter your number of emergency", min_value=0, max_value=100, value=0, step=1, help = "Values from 0 to 76")
    st.session_state.number_inpatient = st.number_input("Enter your number of in patient", min_value=0, max_value=100, value=0, step=1, help = "Values from 0 to 21")
    #st.session_state.diag_1 = st.number_input("Enter your value for number of diagnoses ", min_value=0, max_value=100, value=0, step=1, help = "")
    #st.session_state.diag_2 = st.number_input("Enter your value for diagnoses two", min_value=0, max_value=100, value=0, step=1, help = "")
    #st.session_state.diag_3 = st.number_input("Enter your value for diagnoses three", min_value=0, max_value=100, value=0, step=1, help = "")  

    st.session_state.number_diagnoses = st.number_input("Enter your number of diagnoses", min_value=0, max_value=100, value=8, step=1, help = "Values from 1 to 16")
    #st.session_state.max_glu_serum = st.number_input("Enter your max_glu_serum", min_value=0, max_value=100, value=0, step=1, help = "Range and their relevant value to enter: \n\n >200: 1 \n\n >300: 2 \n\n None: 3 \n\n Norm: 4")
    #st.session_state.A1Cresult = st.number_input("Enter your A1Cresult", min_value=0, max_value=100, value=0, step=1, help ="Range and their relevant value to enter: \n\n >7: 1 \n\n >8: 2 \n\n None: 3 \n\n Norm: 4")
    st.session_state.metformin = st.number_input("Enter your metformin", min_value=0, max_value=100, value=2, step=1, help = "Condition and their relevant value to enter: \n\n Down: 1 \n\n No: 2 \n\n Steady: 3 \n\n Up: 4")
    st.session_state.repaglinide = st.number_input("Enter your repaglinide", min_value=0, max_value=100, value=2, step=1, help = "Condition and their relevant value to enter: \n\n Down: 1 \n\n No: 2 \n\n Steady: 3 \n\n Up: 4")
    st.session_state.nateglinide = st.number_input("Enter your nateglinide", min_value=0, max_value=100, value=2, step=1, help = "Condition and their relevant value to enter: \n\n Down: 1 \n\n No: 2 \n\n Steady: 3 \n\n Up: 4")
    st.session_state.chlorpropamide = st.number_input("Enter your chlorpropamide", min_value=0, max_value=100, value=2, step=1, help = "Condition and their relevant value to enter: \n\n Down: 1 \n\n No: 2 \n\n Steady: 3 \n\n Up: 4")
    st.session_state.glimepiride = st.number_input("Enter your glimepiride", min_value=0, max_value=100, value=2, step=1, help = "Condition and their relevant value to enter: \n\n Down: 1 \n\n No: 2 \n\n Steady: 3 \n\n Up: 4")
    st.session_state.acetohexamide = st.number_input("Enter your acetohexamide", min_value=0, max_value=100, value=1, step=1, help = "Condition and their relevant value to enter: \n\n No: 1 \n\n Steady: 2")
    st.session_state.glipizide = st.number_input("Enter your glipizide", min_value=0, max_value=100, value=2, step=1, help = "Condition and their relevant value to enter: \n\n Down: 1 \n\n No: 2 \n\n Steady: 3 \n\n Up: 4")
    st.session_state.glyburide = st.number_input("Enter your glyburide", min_value=0, max_value=100, value=2, step=1, help = "Condition and their relevant value to enter: \n\n Down: 1 \n\n No: 2 \n\n Steady: 3 \n\n Up: 4")
    st.session_state.tolbutamide = st.number_input("Enter your tolbutamide", min_value=0, max_value=100, value=1, step=1, help = "Condition and their relevant value to enter: \n\n No: 1 \n\n Steady: 2")
    st.session_state.pioglitazone = st.number_input("Enter your pioglitazone", min_value=0, max_value=100, value=2, step=1, help = "Condition and their relevant value to enter: \n\n Down: 1 \n\n No: 2 \n\n Steady: 3 \n\n Up: 4")
    st.session_state.rosiglitazone = st.number_input("Enter your rosiglitazone", min_value=0, max_value=100, value=2, step=1, help = "Condition and their relevant value to enter: \n\n Down: 1 \n\n No: 2 \n\n Steady: 3 \n\n Up: 4")
    st.session_state.acarbose = st.number_input("Enter your acarbose", min_value=0, max_value=100, value=2, step=1, help = "Condition and their relevant value to enter: \n\n Down: 1 \n\n No: 2 \n\n Steady: 3 \n\n Up: 4")
    st.session_state.miglitol = st.number_input("Enter your miglitol", min_value=0, max_value=100, value=2, step=1, help = "Condition and their relevant value to enter: \n\n Down: 1 \n\n No: 2 \n\n Steady: 3 \n\n Up: 4")
    st.session_state.troglitazone = st.number_input("Enter your troglitazone", min_value=0, max_value=100, value=1, step=1, help = "Condition and their relevant value to enter: \n\n No: 1 \n\n Steady: 2")
    st.session_state.tolazamide = st.number_input("Enter your tolazamide", min_value=0, max_value=100, value=1, step=1, help = "Condition and their relevant value to enter: \n\n No: 1 \n\n Steady: 2 \n\n Up: 3")
    st.session_state.examide = st.number_input("Enter your examide", min_value=0, max_value=100, value=1, step=1, help = "Only No: 1")
    st.session_state.citoglipton = st.number_input("Enter your citoglipton", min_value=0, max_value=100, value=1, step=1, help = "Only No: 1")
    st.session_state.insulin = st.number_input("Enter your insulin", min_value=0, max_value=100, value=2, step=1, help = "Only No: 1")

    st.session_state.glyburide_metformin = st.number_input("Enter your glyburide-metformin", min_value=0, max_value=100, value=2, step=1, help = "Condition and their relevant value to enter: \n\n Down: 1 \n\n No: 2 \n\n Steady: 3 \n\n Up: 4")
    st.session_state.glipizide_metformin = st.number_input("Enter your glipizide-metformin", min_value=0, max_value=100, value=1, step=1, help = "Condition and their relevant value to enter: \n\n No: 1 \n\n Steady: 2")
    st.session_state.glimepiride_pioglitazone = st.number_input("Enter your glimepiride-pioglitazone", min_value=0, max_value=100, value=1, step=1, help = "Condition and their relevant value to enter: \n\n No: 1 \n\n Steady: 2")
    st.session_state.metformin_rosiglitazone = st.number_input("Enter your metformin-rosiglitazone", min_value=0, max_value=100, value=1, step=1, help = "Condition and their relevant value to enter: \n\n No: 1 \n\n Steady: 2")
    st.session_state.metformin_pioglitazone = st.number_input("Enter your metformin-pioglitazone", min_value=0, max_value=100, value=1, step=1, help = "Condition and their relevant value to enter: \n\n No: 1 \n\n Steady: 2")
    st.session_state.change = st.number_input("Enter your change", min_value=0, max_value=100, value=2  , step=1, help = "Condition and their relevant value to enter: \n\n Ch: 1 \n\n No: 2")
    #st.session_state.max_glu_serum = st.number_input("Enter your change", min_value=0, max_value=100, value=0, step=1, help = "No, Ch")
    st.session_state.diabetesMed = st.number_input("Enter your diabetes Medication", min_value=0, max_value=100, value=1, step=1, help = "Range and their relevant value to enter: \n\n >200: 1 \n\n >300: 2 \n\n None: 3 \n\n Norm: 4")
    st.session_state.has_diabetes = st.number_input("Enter your has diabetes Medication", min_value=0, max_value=100, value=0, step=1, help = "Range and their relevant value to enter: \n\n >200: 1 \n\n >300: 2 \n\n None: 3 \n\n Norm: 4")
    st.session_state.num_diagnoses = st.number_input("Enter your num diabetes Medication", min_value=0, max_value=100, value=3, step=1, help = "Range and their relevant value to enter: \n\n >200: 1 \n\n >300: 2 \n\n None: 3 \n\n Norm: 4")

    st.session_state.custom_list = [st.session_state.race, st.session_state.gender, st.session_state.age, st.session_state.time_in_hospital, st.session_state.num_lab_procedures, st.session_state.num_procedures, st.session_state.num_medications, st.session_state.number_outpatient, st.session_state.number_emergency, st.session_state.number_inpatient, st.session_state.number_diagnoses, st.session_state.metformin, st.session_state.repaglinide, st.session_state.nateglinide, st.session_state.chlorpropamide, st.session_state.glimepiride, st.session_state.acetohexamide, st.session_state.glipizide, st.session_state.glyburide, st.session_state.tolbutamide, st.session_state.pioglitazone, st.session_state.rosiglitazone, st.session_state.acarbose, st.session_state.miglitol, st.session_state.troglitazone, st.session_state.tolazamide, st.session_state.examide, st.session_state.citoglipton, st.session_state.insulin, st.session_state.glyburide_metformin, st.session_state.glipizide_metformin, st.session_state.glimepiride_pioglitazone, st.session_state.metformin_rosiglitazone, st.session_state.metformin_pioglitazone, st.session_state.change, st.session_state.diabetesMed, st.session_state.has_diabetes, st.session_state.num_diagnoses]


    st.session_state.model_custom_option = st.selectbox("Select an option to make prediction", ["--Select--", "Logistic Regression", "Decision Tree", "Random Forest Classifier (Recommended)", "Ada Boost", "XGBoost", "Naive Bayes"])

    if st.button("Predict!"):
        st.write("Predicting...")
        if st.session_state.model_custom_option == "Logistic Regression":
            result, result1 = call_cust(st.session_state.custom_list, "models/LR_smote.joblib")
            st.write(result)
            
            st.write("The below results indicate the probability of prediction for readmission ")
            st.write("0 indicates readmission not required within 30 days and 1 indicates readmission required within 30 days")

            st.write(result1)

        if st.session_state.model_custom_option == "Decision Tree":
            result, result1 = call_cust(st.session_state.custom_list, "models/DT_smote.joblib")
            st.write(result)
            
            st.write("The below results indicate the probability of prediction for readmission ")
            st.write("0 indicates readmission not required within 30 days and 1 indicates readmission required within 30 days")

            st.write(result1)
        
        if st.session_state.model_custom_option == "Random Forest Classifier (Recommended)":
            result, result1 = call_cust(st.session_state.custom_list, "models/rm_smote_fit.joblib")
            st.write(result)
            
            st.write("The below results indicate the probability of prediction for readmission ")
            st.write("0 indicates readmission not required within 30 days and 1 indicates readmission required within 30 days")

            st.write(result1)

        if st.session_state.model_custom_option == "Ada Boost":
            result, result1 = call_cust(st.session_state.custom_list, "models/adb_tuned.joblib")
            st.write(result)
            
            st.write("The below results indicate the probability of prediction for readmission ")
            st.write("0 indicates readmission not required within 30 days and 1 indicates readmission required within 30 days")

            st.write(result1)

        if st.session_state.model_custom_option == "XGBoost":
            result, result1 = call_cust(st.session_state.custom_list, "models/xgb_classifier.joblib")
            st.write(result)
            
            st.write("The below results indicate the probability of prediction for readmission ")
            st.write("0 indicates readmission not required within 30 days and 1 indicates readmission required within 30 days")

            st.write(result1)

        if st.session_state.model_custom_option == "Naive Bayes":
            result, result1 = call_cust(st.session_state.custom_list, "models/NB1.joblib")
            st.write(result)
            
            
            st.write("The below results indicate the probability of prediction for readmission ")
            st.write("0 indicates readmission not required within 30 days and 1 indicates readmission required within 30 days")

            st.write(result1)

    suggest_option = st.selectbox("Do you want to see why Radom Forest is suggested?", ["--Select--", "Yes", "No"])

    if suggest_option == "Yes":
        st.image("model_information.jpeg", use_column_width=True, caption="Model Comparison")
        st.write("From the dataset we trained, Random Forest algorithm provided the best possible evaluation metrics which best fitted our model after fine tuning.")