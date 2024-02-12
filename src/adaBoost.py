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


def adaBoost(X, y):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score, precision_score, recall_score

         #perform SMOTE on dataset
    from imblearn.over_sampling import SMOTE
    smt = SMOTE(random_state=20)
    X_resampled, Y_resampled = smt.fit_resample(X, y)
    X_resampled = pd.DataFrame(X_resampled, columns = list(X.columns))
    X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_resampled, Y_resampled, test_size=0.20, random_state=20)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20, shuffle=True)

    adb = AdaBoostClassifier(random_state=20)

    adb.fit(X_train, y_train)
    adb_pred = adb.predict(X_test)
    adb_train_accuracy = adb.score(X_train, y_train)
    adb_test_accuracy = adb.score(X_test, y_test)

    st.write(f"<p><strong>Training accuracy:</strong> {adb_train_accuracy:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Testing accuracy:</strong> {adb_test_accuracy:.2f}</p>", unsafe_allow_html=True)

    classification = classification_report(y_test, adb_pred)
    accuracy_adb = accuracy_score(y_test, adb_pred)
    precision_adb = precision_score(y_test, adb_pred)
    recall_adb = recall_score(y_test, adb_pred)


    st.markdown("<h2 style='text-align: center;color: #5fb4fb;'><u>CLASSIFICATION REPORT</u></h2>", unsafe_allow_html=True)

    st.text(classification)

    st.markdown('<hr style="border-top: 1px solid; width: 75%; margin: 0 auto;">', unsafe_allow_html=True)

    st.markdown('<h3 style="color: #5fb4fb;">METRICS:</h3>', unsafe_allow_html=True)

    st.write(f"<p><strong>Accuracy:</strong> {accuracy_adb:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Precision:</strong> {precision_adb:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Recall:</strong> {recall_adb:.2f}</p>", unsafe_allow_html=True)



    from sklearn.model_selection import learning_curve
    import numpy as np
    import matplotlib.pyplot as plt

    adb = AdaBoostClassifier(random_state=20)

    train_sizes, train_scores, test_scores = learning_curve(estimator=adb, X=X_train, y=y_train, cv=5, n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    ax.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    ax.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Cross-validation Accuracy')
    ax.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    ax.set_xlabel('Number of training samples')
    ax.set_ylabel('Accuracy')
    ax.set_title('Learning Curve for Adaboost')
    ax.legend(loc='lower right')

    plt.savefig('learning_curve_adaboost.png')  #Save the plot as an image

    cnf_matrix = metrics.confusion_matrix(y_test, adb_pred)

    class_names=[0,1] 
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    plt.savefig('confusion_matrix_adaboost.png')  #Save the plot as an image


    coll1, coll2 = st.columns([1, 1])
       #First Image
    with coll1:
        st.image("learning_curve_adaboost.png", use_column_width=True, caption="Learning Curve")

    #Second Image
    with coll2:
        st.image("confusion_matrix_adaboost.png", use_column_width=True, caption="Confusion Matrix")


    import plotly.express as px

    df = pd.DataFrame({'Training Accuracy': train_mean, 'Cross-validation Accuracy': test_mean}, index=train_sizes)

    fig = px.line(df, labels={'value': 'Accuracy', 'variable': 'Accuracy Type'}, title='Learning Curve for Adaboost')
    fig.update_traces(error_y=dict(type='data', array=list(test_std), visible=True))
    fig.update_layout(xaxis_title='Number of training samples', yaxis_title='Accuracy')

    st.plotly_chart(fig, use_container_width=True)


    import os
    os.remove("learning_curve_adaboost.png")
    os.remove("confusion_matrix_adaboost.png")


    #rf_option = st.radio("Do you want to perfrom AdaBoost with hyperparameter tuning:", ["Yes", "No"])           
    #if st.button("Select"):
        #if rf_option == "Yes":
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.metrics import classification_report

    adb_tuned = AdaBoostClassifier(random_state=20, n_estimators=73, learning_rate=0.01)

    adb_tuned.fit(X_train_smote, y_train_smote)
    adb_tuned_pred = adb_tuned.predict(X_test_smote)
    adb_train_accuracy_tuned = adb_tuned.score(X_train_smote, y_train_smote)
    adb_test_accuracy_tuned = adb_tuned.score(X_test_smote, y_test_smote)

    st.write(f"<p><strong>Training accuracy:</strong> {adb_train_accuracy_tuned:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Testing accuracy:</strong> {adb_test_accuracy_tuned:.2f}</p>", unsafe_allow_html=True)

    classification_tuned = classification_report(y_test_smote, adb_tuned_pred)

    accuracy_adb1 = accuracy_score(y_test_smote, adb_tuned_pred)
    precision_adb1 = precision_score(y_test_smote, adb_tuned_pred)
    recall_adb1 = recall_score(y_test_smote, adb_tuned_pred)

    st.markdown("<h2 style='text-align: center;color: #5fb4fb;'><u>CLASSIFICATION REPORT</u></h2>", unsafe_allow_html=True)

    st.text(classification_tuned)

    st.markdown('<hr style="border-top: 1px solid; width: 75%; margin: 0 auto;">', unsafe_allow_html=True)

    st.markdown('<h3 style="color: #5fb4fb;">METRICS:</h3>', unsafe_allow_html=True)

    st.write(f"<p><strong>Accuracy:</strong> {accuracy_adb1:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Precision:</strong> {precision_adb1:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Recall:</strong> {recall_adb1:.2f}</p>", unsafe_allow_html=True)

    cnf_matrix = metrics.confusion_matrix(y_test_smote, adb_tuned_pred)

    class_names=[0,1] 
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    plt.savefig('confusion_matrix_adaboost_tuned.png')  #Save the plot as an image

    from sklearn.model_selection import learning_curve

    adb_tuned = AdaBoostClassifier(random_state=20, n_estimators=73, learning_rate=0.01)

    train_sizes, train_scores, test_scores = learning_curve(estimator=adb_tuned, X=X_train_smote, y=y_train_smote, cv=5, n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    ax.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    ax.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Cross-validation Accuracy')
    ax.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    ax.set_xlabel('Number of training samples')
    ax.set_ylabel('Accuracy')
    ax.set_title('Learning Curve for Adaboost')
    ax.legend(loc='lower right')

    plt.savefig('learning_curve_adaboost_tuned.png')  #Save the plot as an image

    precision, recall, accuracy = [precision_adb, precision_adb1], [recall_adb, recall_adb1], [accuracy_adb, accuracy_adb1]

    labels = ['AdaBoost','AdaBoost - fine tuned']

    X_axis = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.bar(labels, precision, width=0.25, label='Precision')
    ax.bar(X_axis+0.25, recall, width=0.25, label='Recall')
    ax.bar(X_axis+0.5, accuracy, width=0.25, label='Accuracy')

    ax.set_xlabel('AdaBoost')
    ax.set_ylabel('Score')
    ax.set_title('AdaBoost Scores')
    ax.legend()

    plt.savefig('adaboost_scores.png')  #Save the plot as an image

    col1, col2 = st.columns([1, 1])

    #First Image
    with col1:
        st.image("confusion_matrix_adaboost_tuned.png", use_column_width=True, caption="Confusion Matrix Adaboost Tuned")

    #Second Image
    with col2:
        st.image("learning_curve_adaboost_tuned.png", use_column_width=True, caption="Learning Curve Adaboost Tuned")

    #Third Image
    with st.container():
        st.image("adaboost_scores.png", width=350, caption="AdaaBoost Curve")

    import plotly.express as px

    dff = pd.DataFrame({'Training Accuracy': train_mean, 'Cross-validation Accuracy': test_mean}, index=train_sizes)

    fig = px.line(dff, labels={'value': 'Accuracy', 'variable': 'Accuracy Type'}, title='Learning Curve for Adaboost (Tuned)')
    fig.update_traces(error_y=dict(type='data', array=list(test_std), visible=True))
    fig.update_layout(xaxis_title='Number of training samples', yaxis_title='Accuracy')

    st.plotly_chart(fig, use_container_width=True)


    #Remove  saved images from disk
    import os
    os.remove("confusion_matrix_adaboost_tuned.png")
    os.remove("learning_curve_adaboost_tuned.png")
    os.remove("adaboost_scores.png") 