import streamlit as st
import pandas as pd
import numpy as np
#from tqdm import tqdm
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

def logisticRegression(X, y):
    #st.write("Entered logisticRegression")
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

     #perform SMOTE on dataset
    from imblearn.over_sampling import SMOTE

    #st.write("0")
    smt = SMOTE(random_state=20)
    #st.write("00")
    X_resampled, Y_resampled = smt.fit_resample(X, y)
    #st.write("000")
    X_resampled = pd.DataFrame(X_resampled, columns = list(X.columns))
    X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_resampled, Y_resampled, test_size=0.20, random_state=20)  

    #st.write("1")
    #st.write(X.head())
    #st.write(y.head())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20, shuffle=True)
    LR = LogisticRegression()
    LR.fit(X_train, y_train)
    LR_pred = LR.predict(X_test)
    #st.write("2")
    classification_report_str = classification_report(y_test, LR_pred)
    accuracy_lr = accuracy_score(y_test, LR_pred)
    precision_lr = precision_score(y_test, LR_pred)
    recall_lr = recall_score(y_test, LR_pred)

    st.markdown("<h2 style='text-align: center;color: #5fb4fb;'><u>CLASSIFICATION REPORT</u></h2>", unsafe_allow_html=True)
    st.text(classification_report_str)

    st.markdown('<hr style="border-top: 1px solid; width: 75%; margin: 0 auto;">', unsafe_allow_html=True)

    st.markdown('<h3 style="color: #5fb4fb;">METRICS:</h3>', unsafe_allow_html=True)
    st.write(f"<p><strong>Accuracy:</strong> {accuracy_lr:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Precision:</strong> {precision_lr:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Recall:</strong> {recall_lr:.2f}</p>", unsafe_allow_html=True)

    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix

    precision, recall, _ = precision_recall_curve(y_test, LR_pred)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label='Precision-recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.savefig("precision_recall_curve.png")
    plt.close()  # Close the current figure to release memory

    fpr, tpr, _ = roc_curve(y_test, LR_pred)
    auc = roc_auc_score(y_test, LR_pred)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)

    plt.savefig("roc_curve.png")
    plt.close()  # Close the current figure to release memory

    cnf_matrix = confusion_matrix(y_test, LR_pred)
    class_names = [0, 1]
    fig, ax = plt.subplots(figsize=(6, 4))
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion Matrix', y=1.1)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    plt.savefig("confusion_matrix.png")
    plt.close()

    col1, col2 = st.columns([1, 1]) 

    #First Image
    with col1:
        st.image("precision_recall_curve.png", use_column_width=True, caption="Precision-Recall Curve")

    #Second Image
    with col2:
        st.image("roc_curve.png", use_column_width=True, caption="ROC Curve")

    #Third Image
    with st.container():
        st.image("confusion_matrix.png", width=350, caption="Confusion Matrix")

    #Remove saved images from disk
    import os
    os.remove("precision_recall_curve.png")
    os.remove("roc_curve.png")
    os.remove("confusion_matrix.png")


    #lr_option = st.radio("Do you want to perform LR using weight balanced:", ["Yes", "No"], index = None)
    #select_button1 = st.button("Select", key="select_button1")
    #if submit_button:
        #if lr_option == "Yes":

    #lr_option = st.selectbox("Do you want to perform LR using weight balanced?", ["No", "Yes"])

    #if lr_option == "Yes":

    st.write("Performing LR using weight balancing")
    LR1 = LogisticRegression(solver='sag',class_weight='balanced')
    LR1.fit(X_train, y_train)
    LR1_pred = LR1.predict(X_test)

    accuracy_lr1 = accuracy_score(y_test, LR1_pred)
    precision_lr1 = precision_score(y_test, LR1_pred)
    recall_lr1 = recall_score(y_test, LR1_pred)

    classification_report_str1 = classification_report(y_test, LR1_pred)
    accuracy_lr1 = accuracy_score(y_test, LR1_pred)
    precision_lr1 = precision_score(y_test, LR1_pred)
    recall_lr1 = recall_score(y_test, LR1_pred)

    st.markdown("<h2 style='text-align: center;color: #5fb4fb;'><u>CLASSIFICATION REPORT</u></h2>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>LR using weight balanced</h5>", unsafe_allow_html=True)
    st.text(classification_report_str1)

    st.markdown('<hr style="border-top: 1px solid; width: 75%; margin: 0 auto;">', unsafe_allow_html=True)
    
    st.markdown('<h3 style="color: #5fb4fb;">METRICS:</h3>', unsafe_allow_html=True)
    st.write(f"<p><strong>Accuracy:</strong> {accuracy_lr1:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Precision:</strong> {precision_lr1:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Recall:</strong> {recall_lr1:.2f}</p>", unsafe_allow_html=True)


    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix

    fpr, tpr, _ = roc_curve(y_test, LR1_pred)
    auc = roc_auc_score(y_test, LR1_pred)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    ax.legend(loc=4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')

    plt.savefig("roc_curve.png")

    precision, recall, _ = precision_recall_curve(y_test, LR1_pred)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label='Precision-recall curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-recall curve')
    ax.legend(loc='lower left')

    plt.savefig("precision_recall_curve.png")

    cnf_matrix = confusion_matrix(y_test, LR1_pred)
    class_names = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    plt.savefig("confusion_matrix.png")


    col1, col2 = st.columns([1, 1]) 

    #First Image
    with col1:
        st.image("precision_recall_curve.png", use_column_width=True, caption="Precision-Recall Curve")

    #Second Image
    with col2:
        st.image("roc_curve.png", use_column_width=True, caption="ROC Curve")

    #Third Image
    with st.container():
        st.image("confusion_matrix.png", width=350, caption="Confusion Matrix")

    #Remove saved images from disk
    import os
    os.remove("precision_recall_curve.png")
    os.remove("roc_curve.png")
    os.remove("confusion_matrix.png")
    
    #lr_option2 = st.radio("Do you want to perform LR using SMOTE:", ["Yes", "No"])
    #select_button2 = st.button("Select", key="select_button2")
    #if submit_button2:
        #if lr_option2 == "Yes":

    #lr_option2 = st.selectbox("Do you want to perform LR using SMOTE?", ["No", "Yes"])

    #if lr_option2 == "Yes":

    st.write("LR using SMOTE and weight balancing")
    LR2 = LogisticRegression(solver='sag',class_weight='balanced')
    LR2_fit = LR2.fit(X_train_smote, y_train_smote)
    LR2_pred_smote = LR2_fit.predict(X_test_smote)

    classification_rep = classification_report(y_test_smote, LR2_pred_smote)
    accuracy_lr2 = accuracy_score(y_test_smote, LR2_pred_smote)
    precision_lr2 = precision_score(y_test_smote, LR2_pred_smote)
    recall_lr2 = recall_score(y_test_smote, LR2_pred_smote)

    st.markdown("<h2 style='text-align: center;color: #5fb4fb;'><u>CLASSIFICATION REPORT</u></h2>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>LR using SMOTE</h5>", unsafe_allow_html=True)
    st.text(classification_rep)

    st.markdown('<hr style="border-top: 1px solid; width: 75%; margin: 0 auto;">', unsafe_allow_html=True)

    st.markdown('<h3 style="color: #5fb4fb;">METRICS:</h3>', unsafe_allow_html=True)
    st.write(f"<p><strong>Accuracy:</strong> {accuracy_lr2:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Precision:</strong> {precision_lr2:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Recall:</strong> {recall_lr2:.2f}</p>", unsafe_allow_html=True)

    

    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_test_smote, LR2_pred_smote)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, label='Precision-recall curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-recall curve')
    ax.legend(loc='lower left')

    plt.savefig("precision_recall_curve.png")

    from sklearn import metrics

    fpr, tpr, _ = roc_curve(y_test_smote, LR2_pred_smote)
    auc = roc_auc_score(y_test_smote, LR2_pred_smote)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="auc" + str(auc))
    ax.legend(loc=4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')

    plt.savefig("roc_curve.png")

    cnf_matrix = confusion_matrix(y_test_smote, LR2_pred_smote)
    class_names = [0, 1]

    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    plt.savefig("confusion_matrix.png")

    precision = [precision_lr, precision_lr1, precision_lr2]
    recall = [recall_lr, recall_lr1, recall_lr2]
    accuracy = [accuracy_lr, accuracy_lr1, accuracy_lr2]
    labels = ['Logistic Regression', 'LR - weight balanced', 'LR - SMOTE - WB']

    X_axis = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.bar(labels, precision, width=0.25, label='Precision')
    ax.bar(X_axis + 0.25, recall, width=0.25, label='Recall')
    ax.bar(X_axis + 0.5, accuracy, width=0.25, label='Accuracy')

    ax.set_xlabel('Logistic Regression')
    ax.set_ylabel('Score')
    ax.set_title('LR Scores')
    ax.legend()

    plt.savefig("lr_scores.png")

    col1, col2 = st.columns([1, 1])

    #First Image
    with col1:
        st.image("precision_recall_curve.png", use_column_width=True, caption="Precision-Recall Curve")

    #Second Image
    with col2:
        st.image("roc_curve.png", use_column_width=True, caption="ROC Curve")

    coll1, coll2 = st.columns([1, 1])
    #Third Image
    with coll1:
        st.image("confusion_matrix.png", use_column_width=True, caption="Confusion Matrix")

    with coll2:
        st.image("lr_scores.png", use_column_width=True, caption="LR Scores")


    import plotly.express as px
    dff = pd.DataFrame({'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}, index=labels)

    fig = px.bar(dff, barmode='group')
    fig.update_layout(title='Interactive LR Scores', xaxis_title='Logistic Regression', yaxis_title='Score')

    st.plotly_chart(fig, use_container_width=True)


    #Remove saved images from disk
    import os
    os.remove("precision_recall_curve.png")
    os.remove("roc_curve.png")
    os.remove("confusion_matrix.png")
    os.remove("lr_scores.png")
