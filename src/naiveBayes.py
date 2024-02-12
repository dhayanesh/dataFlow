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

def naiveBayes(X, y):

     #perform SMOTE on dataset
    from imblearn.over_sampling import SMOTE
    smt = SMOTE(random_state=20)
    X_resampled, Y_resampled = smt.fit_resample(X, y)
    X_resampled = pd.DataFrame(X_resampled, columns = list(X.columns))
    X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_resampled, Y_resampled, test_size=0.20, random_state=20)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20, shuffle=True)
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.metrics import accuracy_score, precision_score, recall_score


    NB=BernoulliNB()
    NB.fit(X_train,y_train)
    NB_pred=NB.predict(X_test)
    NB_training_accuracy = NB.score(X_train,y_train)
    NB_testing_accuracy = accuracy_score(y_test,NB_pred)

    st.write(f"<p><strong>Training Accuracy:</strong> {NB_training_accuracy:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Testing Accuracy:</strong> {NB_testing_accuracy:.2f}</p>", unsafe_allow_html=True)

    accuracy_nb = accuracy_score(y_test, NB_pred)
    precision_nb = precision_score(y_test, NB_pred)
    recall_nb = recall_score(y_test, NB_pred)

    st.markdown("<h2 style='text-align: center;color: #5fb4fb;'><u>CLASSIFICATION REPORT</u></h2>", unsafe_allow_html=True)

    st.text(classification_report(y_test,NB_pred))

    st.markdown('<hr style="border-top: 1px solid; width: 75%; margin: 0 auto;">', unsafe_allow_html=True)

    st.markdown('<h3 style="color: #5fb4fb;">METRICS:</h3>', unsafe_allow_html=True)

    st.write(f"<p><strong>Accuracy:</strong> {accuracy_nb:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Precision:</strong> {precision_nb:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Recall:</strong> {recall_nb:.2f}</p>", unsafe_allow_html=True)



    from sklearn.metrics import confusion_matrix

    cnf_matrix = confusion_matrix(y_test, NB_pred)

    sns.heatmap(cnf_matrix, cmap='Blues', annot=True, fmt='g')

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    #st.pyplot()

    import plotly.figure_factory as ff
    from sklearn.metrics import confusion_matrix


    cnf_matrix = confusion_matrix(y_test, NB_pred)

    normalized_matrix = np.round(cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis], decimals=2)

    fig = ff.create_annotated_heatmap(z=normalized_matrix, x=['No', 'Yes'], y=['No', 'Yes'], colorscale='Blues')
    fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted labels', yaxis_title='True labels')

    st.plotly_chart(fig, use_container_width=True)


    #nb_option = st.radio("Do you want to perfrom Naive Bayes with SMOTE:", ["Yes", "No"])           
    #if st.button("Select"):
        #if nb_option == "Yes":

    NB1=BernoulliNB()
    NB1.fit(X_train_smote,y_train_smote)
    NB_pred1=NB1.predict(X_test_smote)
    NB_training_accuracy = NB1.score(X_train_smote,y_train_smote)
    NB_testing_accuracy = accuracy_score(y_test_smote,NB_pred1)

    st.write(f"<p><strong>Training Accuracy:</strong> {NB_training_accuracy:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Testing Accuracy:</strong> {NB_testing_accuracy:.2f}</p>", unsafe_allow_html=True)

    accuracy_nb1 = accuracy_score(y_test_smote,NB_pred1)
    precision_nb1 = precision_score(y_test_smote,NB_pred1)
    recall_nb1 = recall_score(y_test_smote,NB_pred1)

    st.markdown("<h2 style='text-align: center;color: #5fb4fb;'><u>CLASSIFICATION REPORT</u></h2>", unsafe_allow_html=True)
    st.markdown('<h3>Naive Bayes with SMOTE</h3>', unsafe_allow_html=True)
    st.text(classification_report(y_test_smote,NB_pred1))

    st.markdown('<hr style="border-top: 1px solid; width: 75%; margin: 0 auto;">', unsafe_allow_html=True)

    st.markdown('<h3 style="color: #5fb4fb;">METRICS:</h3>', unsafe_allow_html=True)

    st.write(f"<p><strong>Accuracy:</strong> {accuracy_nb1:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Precision:</strong> {precision_nb1:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Recall:</strong> {recall_nb1:.2f}</p>", unsafe_allow_html=True)

    from sklearn import metrics
    cnf_matrix = metrics.confusion_matrix(y_test_smote, NB_pred1)

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

    plt.savefig('confusion_matrix_nb.png')  #Save the plot as an image

    precision, recall, accuracy = [precision_nb, precision_nb1], [recall_nb, recall_nb1], [accuracy_nb, accuracy_nb1]

    labels = ['Naive Bayes','NB - fine tuned']

    X_axis = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.bar(labels, precision, width=0.25, label='Precision')
    ax.bar(X_axis+0.25, recall, width=0.25, label='Recall')
    ax.bar(X_axis+0.5, accuracy, width=0.25, label='Accuracy')

    ax.set_xlabel('Naive Bayes')
    ax.set_ylabel('Score')
    ax.set_title('Naive Bayes Scores')
    ax.legend()

    plt.savefig('naive_bayes_scores.png')  #Save the plot as an image

    coll1, coll2 = st.columns([1, 1])

    with coll1:
        st.image("confusion_matrix_nb.png", width=350, caption="Confusion Matrix")
    with coll2:
        st.image("naive_bayes_scores.png", width=350, caption="Caliberation Plot")


    import plotly.graph_objects as go

    data = [
        go.Bar(name='Precision', x=['Naive Bayes', 'NB - fine tuned'], y=[precision_nb, precision_nb1]),
        go.Bar(name='Recall', x=['Naive Bayes', 'NB - fine tuned'], y=[recall_nb, recall_nb1]),
        go.Bar(name='Accuracy', x=['Naive Bayes', 'NB - fine tuned'], y=[accuracy_nb, accuracy_nb1])
    ]

    layout = go.Layout(title='Naive Bayes Scores', xaxis_title='Naive Bayes', yaxis_title='Score')

    fig = go.Figure(data=data, layout=layout)

    st.plotly_chart(fig, use_container_width=True)


    #Remove the saved images from disk
    import os
    os.remove("confusion_matrix_nb.png")
    os.remove("naive_bayes_scores.png")