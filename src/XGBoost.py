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


def XGBoost(X, y):

    #perform SMOTE on dataset
    from imblearn.over_sampling import SMOTE
    smt = SMOTE(random_state=20)
    X_resampled, Y_resampled = smt.fit_resample(st.session_state.X, st.session_state.y)
    X_resampled = pd.DataFrame(X_resampled, columns = list(X.columns))
    X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_resampled, Y_resampled, test_size=0.20, random_state=20)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20, shuffle=True)
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, classification_report

    xgb_classifier = xgb.XGBClassifier()

    xgb_classifier.fit(X_train, y_train)

    y_pred_xgb = xgb_classifier.predict(X_test)

    Train_Score_xgb = xgb_classifier.score(X_train, y_train)
    Test_Score_xgb = accuracy_score(y_test, y_pred_xgb)

    st.write(f"<p><strong>Training Accuracy:</strong> {Train_Score_xgb:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Testing Accuracy:</strong> {Test_Score_xgb:.2f}</p>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center;color: #5fb4fb;'><u>CLASSIFICATION REPORT</u></h2>", unsafe_allow_html=True)

    st.text(classification_report(y_test, y_pred_xgb))

    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_test, y_pred_xgb)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, label='Precision-recall curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-recall curve')
    ax.legend(loc="lower left")

    plt.savefig('precision_recall_curve.png')  #Save the plot as an image

    from sklearn import metrics

    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_xgb)
    auc = metrics.roc_auc_score(y_test, y_pred_xgb)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="XG boost curve, auc="+str(auc))
    ax.legend(loc=4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')

    plt.savefig('roc_curve.png')  #Save the plot as an image

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred_xgb)

    class_names=[0,1] 
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sb.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    plt.savefig('confusion_matrix.png')  #Save the plot as an image

    from sklearn.metrics import classification_report
    import plotly.figure_factory as ff

    report = classification_report(y_test, y_pred_xgb, output_dict=True)
    fig = ff.create_annotated_heatmap(z=[[report['0']['precision'], report['0']['recall'], report['0']['f1-score']],
                                        [report['1']['precision'], report['1']['recall'], report['1']['f1-score']]],
                                    x=['precision', 'recall', 'f1-score'],
                                    y=['Negative', 'Positive'], colorscale='Viridis')
    fig.update_layout(title='Classification Report Heat Map')

    fig.write_image("classification_report_heatmap.png")  #Save the plot as an image
    st.image("classification_report_heatmap.png", use_column_width=True, caption="Classification Report Heatmap")

    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_test, y_pred_xgb, n_bins=10)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    ax.plot(prob_pred, prob_true, marker='.', label='XGBoost')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('True probability')
    ax.set_title('Calibration plot')
    ax.legend()

    plt.savefig('calibration_plot.png')  #Save the plot as an image



    col1, col2 = st.columns([1, 1])

    #First Image
    with col1:
        st.image("precision_recall_curve.png", use_column_width=True, caption="Precison Recall Curve")

    #Second Image
    with col2:
        st.image("roc_curve.png", use_column_width=True, caption="ROC Curve")

    coll1, coll2 = st.columns([1, 1])

    #Third Image
    with coll1:
        st.image("confusion_matrix.png", width=350, caption="Confusion Matrix")
    with coll2:
        st.image("calibration_plot.png", width=350, caption="Caliberation Plot")


    import plotly.express as px

    df = pd.DataFrame({'Predicted Probability': prob_pred, 'True Probability': prob_true})

    fig = px.scatter(df, x='Predicted Probability', y='True Probability', title='Interactive Calibration plot', trendline='ols')
    fig.update_layout(xaxis_title='Predicted probability', yaxis_title='True probability')

    st.plotly_chart(fig, use_container_width=True)


    #Remove  saved images from disk
    import os
    os.remove("precision_recall_curve.png")
    os.remove("roc_curve.png")
    os.remove("confusion_matrix.png") 
    os.remove("calibration_plot.png") 
    os.remove("classification_report_heatmap.png") 
    



    #xg_option = st.radio("Do you want to perfrom XGBoost with hyperparameter tuning:", ["Yes", "No"])           
    #if st.button("Select"):
        #if xg_option == "Yes":
    
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, classification_report

    xgb_classifier = xgb.XGBClassifier()

    
    xgb_classifier.fit(X_train_smote, y_train_smote)

    y_preds_xgb = xgb_classifier.predict(X_test_smote)

    Trains_Score_xgb = xgb_classifier.score(X_train_smote, y_train_smote)
    Tests_Score_xgb = accuracy_score(y_test_smote, y_preds_xgb)

    #st.write('Training Accuracy is:', Trains_Score_xgb)
    #st.write('Testing Accuracy is:', Tests_Score_xgb)


    st.write(f"<p><strong>Training Accuracy:</strong> {Trains_Score_xgb:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Testing Accuracy:</strong> {Tests_Score_xgb:.2f}</p>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center;color: #5fb4fb;'><u>CLASSIFICATION REPORT</u></h2>", unsafe_allow_html=True)


    st.text(classification_report(y_test_smote, y_preds_xgb))

    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_test_smote, y_preds_xgb)

    plt.plot(recall, precision, label='Precision-recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-recall curve')
    plt.legend(loc="lower left")

    plt.savefig('precision_recall_curve.png')  #Save the plot as an image

    from sklearn import metrics
    import matplotlib.pyplot as plt

    fpr, tpr, _ = metrics.roc_curve(y_test_smote,  y_preds_xgb)
    auc = metrics.roc_auc_score(y_test_smote, y_preds_xgb)
    plt.plot(fpr, tpr, label="XG boost SMOTE, auc="+str(auc))
    plt.legend(loc=4)

    plt.savefig('roc_curve.png')  #Save the plot as an image

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred_xgb)

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

    plt.savefig('confusion_matrix.png')  #Save the plot as an image

    import plotly.figure_factory as ff
    from sklearn.metrics import classification_report

    report = classification_report(y_test_smote, y_preds_xgb, output_dict=True)
    fig = ff.create_annotated_heatmap(z=[[report['0']['precision'], report['0']['recall'], report['0']['f1-score']],
                                        [report['1']['precision'], report['1']['recall'], report['1']['f1-score']]],
                                    x=['precision', 'recall', 'f1-score'],
                                    y=['Negative', 'Positive'], colorscale='Viridis')
    fig.update_layout(title='Classification Report Heat Map')
    fig.write_image("classification_report_heatmap.png")  #Save the plot as an image

    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_test_smote, y_preds_xgb, n_bins=10)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    ax.plot(prob_pred, prob_true, marker='.', label='XGBoost')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('True probability')
    ax.set_title('Calibration plot')
    ax.legend()

    plt.savefig('calibration_plot.png')  #Save the plot as an image


    col1, col2 = st.columns([1, 1])

    #First Image
    with col1:
        st.image("precision_recall_curve.png", use_column_width=True, caption="Precison Recall Curve")

    #Second Image
    with col2:
        st.image("roc_curve.png", use_column_width=True, caption="ROC Curve")

    coll1, coll2 = st.columns([1, 1])

    #Third Image
    with coll1:
        st.image("confusion_matrix.png", width=350, caption="Confusion Matrix")
    with coll2:
        st.image("calibration_plot.png", width=350, caption="Caliberation Plot")


    import plotly.express as px


    df = pd.DataFrame({'Predicted Probability': prob_pred, 'True Probability': prob_true})

    fig = px.scatter(df, x='Predicted Probability', y='True Probability', title='Interactive Calibration plot', trendline='ols')
    fig.update_layout(xaxis_title='Predicted probability', yaxis_title='True probability')

    st.plotly_chart(fig, use_container_width=True)


    #Remove the saved images from disk
    import os
    os.remove("precision_recall_curve.png")
    os.remove("roc_curve.png")
    os.remove("confusion_matrix.png") 
    os.remove("calibration_plot.png") 
   