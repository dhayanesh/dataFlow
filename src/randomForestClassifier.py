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

def randomForestClassifier(X, y):

     #perform SMOTE on dataset
    from imblearn.over_sampling import SMOTE
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_curve, roc_auc_score, confusion_matrix

    smt = SMOTE(random_state=20)
    X_resampled, Y_resampled = smt.fit_resample(X, y)
    X_resampled = pd.DataFrame(X_resampled, columns = list(X.columns))
    X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_resampled, Y_resampled, test_size=0.20, random_state=20)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20, shuffle=True)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

    rm = RandomForestClassifier(n_estimators=10, max_depth=15, criterion="gini", min_samples_split=10)
    rm.fit(X_train, y_train)
    rm_pred = rm.predict(X_test)

    st.markdown("<h2 style='text-align: center;color: #5fb4fb;'><u>CLASSIFICATION REPORT</u></h2>", unsafe_allow_html=True)

    st.text(classification_report(y_test, rm_pred))

    st.markdown('<hr style="border-top: 1px solid; width: 75%; margin: 0 auto;">', unsafe_allow_html=True)

    st.markdown('<h3 style="color: #5fb4fb;">METRICS:</h3>', unsafe_allow_html=True)

    accuracy_rm = accuracy_score(y_test, rm_pred)
    precision_rm = precision_score(y_test, rm_pred)
    recall_rm = recall_score(y_test, rm_pred)

    st.write(f"<p><strong>Accuracyg:</strong> {accuracy_rm:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Precision:</strong> {precision_rm:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Recall:</strong> {recall_rm:.2f}</p>", unsafe_allow_html=True)


    from sklearn.metrics import roc_curve, auc

    rm = RandomForestClassifier(n_estimators=10, max_depth=15, criterion="gini", min_samples_split=10)
    rm.fit(X_train, y_train)
    rm_pred = rm.predict(X_test)

    y_score = rm.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, rm_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Random Forest')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_rf.png')  #Save the plot as an image

    cnf_matrix = confusion_matrix(y_test, rm_pred)

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
    plt.savefig('confusion_matrix_rf.png')  #Save the plot as an image

    fpr, tpr, thresholds = roc_curve(y_test, rm_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Random Forest')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_rf2.png')  #Save the plot as an image


    col1, col2 = st.columns([1, 1])

    #First Image
    with col1:
        st.image("roc_curve_rf.png", use_column_width=True, caption="ROC Curve")

    #Second Image
    with col2:
        st.image("confusion_matrix_rf.png", use_column_width=True, caption="Confusion Matrix")

    #Third Image
    with st.container():
        st.image("roc_curve_rf2.png", width=350, caption="ROC Curve")

    # Remove the saved images from disk
    import os
    os.remove("roc_curve_rf.png")
    os.remove("confusion_matrix_rf.png")
    os.remove("roc_curve_rf2.png") 


    #rf_option = st.radio("Do you want to perfrom Random Forest with SMOTE:", ["Yes", "No"])           
    #if st.button("Select"):
        #if rf_option == "Yes":
    rm_smote = RandomForestClassifier(n_estimators=10, max_depth=25, criterion="gini", min_samples_split=10)
    rm_smote_fit = rm_smote.fit(X_train_smote, y_train_smote)
    rm_smote_pred = rm_smote_fit.predict(X_test_smote)

    st.markdown("<h2 style='text-align: center;color: #5fb4fb;'><u>CLASSIFICATION REPORT</u></h2>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>Random Forest with SMOTE</h5>", unsafe_allow_html=True)

    st.text(classification_report(y_test_smote, rm_smote_pred))

    st.markdown('<hr style="border-top: 1px solid; width: 75%; margin: 0 auto;">', unsafe_allow_html=True)

    st.markdown('<h3 style="color: #5fb4fb;">METRICS:</h3>', unsafe_allow_html=True)

    accuracy_rm1 = accuracy_score(y_test_smote, rm_smote_pred)
    precision_rm1 = precision_score(y_test_smote, rm_smote_pred)
    recall_rm1 = recall_score(y_test_smote, rm_smote_pred)

    st.write(f"<p><strong>Accuracyg:</strong> {accuracy_rm1:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Precision:</strong> {precision_rm1:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Recall:</strong> {recall_rm1:.2f}</p>", unsafe_allow_html=True)



    from sklearn.metrics import confusion_matrix

    cnf_matrix = confusion_matrix(y_test_smote, rm_smote_pred)

    plt.figure(figsize=(10, 8))
    tick_marks = np.arange(len(['0', '1']))
    plt.xticks(tick_marks, ['0', '1'])
    plt.yticks(tick_marks, ['0', '1'])
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g', cbar=False, square=True)
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    plt.title('Confusion Matrix for Random Forest with SMOTE')
    plt.savefig('confusion_matrix_rf_smote.png')  #Save the plot as an image

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(y_test_smote, rm_smote_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Random Forest with SMOTE')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_rf_smote.png')  #Save the plot as an image


    col1, col2 = st.columns([1, 1])

    #First Image
    with col1:
        st.image("confusion_matrix_rf_smote.png", use_column_width=True, caption="Confusion Matrix")

    #Second Image
    with col2:
        st.image("roc_curve_rf_smote.png", use_column_width=True, caption="ROC Curve")

    #Remove saved images from disk
    import os
    os.remove("confusion_matrix_rf_smote.png")
    os.remove("roc_curve_rf_smote.png")
        
   #rf_option = st.radio("Do you want to perfrom Random Forest with best parameter estimation:", ["Yes", "No"])           
    #if st.button("Select"):
        #if rf_option == "Yes":

    st.write(f"<p style='color:#5fb4fb'><strong>Fine tuning Random Forest with best parameter estimation (Caution: This might take longer to run!)</strong></p>", unsafe_allow_html=True)    

    from sklearn.model_selection import GridSearchCV
    parameter={'n_estimators':np.arange(1,80)}
    gs = GridSearchCV(rm,parameter,cv=3)
    gs.fit(X_train,y_train)

    st.text('Best Hyperparameters:', gs.best_params_)

    rm_finetuned = RandomForestClassifier(n_estimators=66)
    rm_finetuned_fit = rm_finetuned.fit(X_train_smote,y_train_smote) 
    rm_finetuned_pred = rm_finetuned_fit.predict(X_test_smote)
    rm_finetuned_training_accuracy = rm_finetuned_fit.score(X_train_smote,y_train_smote)
    rm_finetuned_testing_accuracy = accuracy_score(y_test_smote,rm_finetuned_pred)

    st.write(f"<p><strong>Training Accuracy:</strong> {rm_finetuned_training_accuracy:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Testing Accuracy:</strong> {rm_finetuned_testing_accuracy:.2f}</p>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center;color: #5fb4fb;'><u>CLASSIFICATION REPORT</u></h2>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>Random Forest with with best parameter estimation</h5>", unsafe_allow_html=True)


    st.text(classification_report(y_test_smote,rm_finetuned_pred))

    st.markdown('<hr style="border-top: 1px solid; width: 75%; margin: 0 auto;">', unsafe_allow_html=True)

    st.markdown('<h3 style="color: #5fb4fb;">METRICS:</h3>', unsafe_allow_html=True)

    accuracy_rm2 = accuracy_score(y_test_smote, rm_finetuned_pred)
    precision_rm2 = precision_score(y_test_smote, rm_finetuned_pred)
    recall_rm2 = recall_score(y_test_smote, rm_finetuned_pred)

    st.write(f"<p><strong>Accuracy:</strong> {accuracy_rm2:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Precision:</strong> {precision_rm2:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Recall:</strong> {recall_rm2:.2f}</p>", unsafe_allow_html=True)

    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

    feature_importances = rm_finetuned.feature_importances_
    feature_names = X.columns.tolist()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(feature_names, feature_importances)
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Feature Name')
    ax.set_title('Feature Importance for Random Forest')
    plt.savefig('feature_importance_rf.png')  #Save the plot as an image

    y_score = rm_finetuned.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test_smote, rm_finetuned_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Random Forest')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_rf.png')  #Save the plot as an image

    cnf_matrix = confusion_matrix(y_test_smote, rm_finetuned_pred)

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
    plt.savefig('confusion_matrix_rf.png')  #Save the plot as an image

    fpr, tpr, _ = roc_curve(y_test_smote, rm_finetuned_pred)
    auc = roc_auc_score(y_test_smote, rm_finetuned_pred)

    fig, ax = plt.subplots()
    plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('roc_curve_rf2.png')  #Save the plot as an image


    col1, col2 = st.columns([1, 1])

    #First Image
    with col1:
        st.image("feature_importance_rf.png", use_column_width=True, caption="Feature Importance")

    #Second Image
    with col2:
        st.image("roc_curve_rf.png", use_column_width=True, caption="ROC Curve")

    coll1, coll2 = st.columns([1, 1])
    with coll1:
        st.image("confusion_matrix_rf.png", use_column_width=True, caption="Confusion Matrix")

    with coll2:
        st.image("roc_curve_rf2.png", use_column_width=True, caption="ROC Curve")


    import plotly.graph_objects as go

    feature_importances = rm_finetuned.feature_importances_
    feature_names = X.columns.tolist()

    fig = go.Figure(go.Bar(
                x=feature_importances,
                y=feature_names,
                orientation='h'))

    fig.update_layout(title='Interactive Feature Importance for Random Forest',
                    xaxis_title='Feature Importance',
                    yaxis_title='Feature Name')

    st.plotly_chart(fig, use_container_width=True)


    import os
    os.remove("feature_importance_rf.png")
    os.remove("roc_curve_rf.png")
    os.remove("confusion_matrix_rf.png")
    os.remove("roc_curve_rf2.png")


    import matplotlib.pyplot as plt

    precision, recall, accuracy = [precision_rm, precision_rm1, precision_rm2], [recall_rm, recall_rm1, recall_rm2], [accuracy_rm, accuracy_rm1, accuracy_rm2]

    labels = ['Random Forest', 'RF - SMOTE', 'RF - fine tuned']

    X_axis = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.bar(labels, precision, width=0.25, label='Precision')
    ax.bar(X_axis+0.25, recall, width=0.25, label='Recall')
    ax.bar(X_axis+0.5, accuracy, width=0.25, label='Accuracy')

    ax.set_xlabel('Random Forest')
    ax.set_ylabel('Score')
    ax.set_title('RF Scores')
    ax.legend()

    st.pyplot(fig)