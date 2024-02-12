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

def decisionTree(X, y):

    #perform SMOTE on dataset
    from imblearn.over_sampling import SMOTE
    smt = SMOTE(random_state=20)
    X_resampled, Y_resampled = smt.fit_resample(X, y)
    X_resampled = pd.DataFrame(X_resampled, columns = list(X.columns))
    X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_resampled, Y_resampled, test_size=0.20, random_state=20)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20, shuffle=True)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

    #create Decision Tree classifier
    DT = DecisionTreeClassifier(max_depth=10, criterion="entropy", min_samples_split=10)
    DT_fit = DT.fit(X_train, y_train)
    DT_pred = DT_fit.predict(X_test)

    st.markdown("<h2 style='text-align: center;color: #5fb4fb;'><u>CLASSIFICATION REPORT</u></h2>", unsafe_allow_html=True)

    st.text(classification_report(y_test, DT_pred))

    st.markdown('<hr style="border-top: 1px solid; width: 75%; margin: 0 auto;">', unsafe_allow_html=True)

    st.markdown('<h3 style="color: #5fb4fb;">METRICS:</h3>', unsafe_allow_html=True)

    accuracy_dt = accuracy_score(y_test, DT_pred)
    precision_dt = precision_score(y_test, DT_pred)
    recall_dt = recall_score(y_test, DT_pred)

    st.write(f"<p><strong>Accuracy:</strong> {accuracy_dt:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Precision:</strong> {precision_dt:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Recall:</strong> {recall_dt:.2f}</p>", unsafe_allow_html=True)


    import matplotlib.pyplot as plt

    feature_importances = DT.feature_importances_
    feature_names = X.columns.tolist()

    plt.figure(figsize=(10, 8))
    plt.barh(feature_names, feature_importances)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title('Feature Importance for Decision Tree')

    plt.savefig("feature_importance.png")

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_test, DT_pred)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Decision Tree initial ROC')
    plt.legend(loc="lower right")

    plt.savefig("roc_curve.png")

    from sklearn import metrics

    cnf_matrix = confusion_matrix(y_test, DT_pred)

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

    plt.savefig("confusion_matrix.png")


    col1, col2 = st.columns([1, 1])

    #First Image
    with col1:
        st.image("feature_importance.png", use_column_width=True, caption="Feature Importance for Decision Tree")

    #Second Image
    with col2:
        st.image("roc_curve.png", use_column_width=True, caption="Decision Tree initial ROC")

    #Third Image
    with st.container():
        st.image("confusion_matrix.png", width=350, caption="Confusion Matrix")


    import plotly.express as px

    df = pd.DataFrame({'Feature Name': feature_names, 'Feature Importance': feature_importances})

    fig = px.bar(df, x='Feature Importance', y='Feature Name', orientation='h', title='Interactive Feature Importance for Decision Tree')
    fig.update_layout(xaxis_title='Feature Importance', yaxis_title='Feature Name')

    st.plotly_chart(fig, use_container_width=True)


    #Remove saved images from disk
    import os
    os.remove("feature_importance.png")
    os.remove("roc_curve.png")
    os.remove("confusion_matrix.png")


    from sklearn.tree import export_graphviz
    import graphviz

    dot_data = export_graphviz(
        DT_fit,
        out_file=None,
        feature_names=X.columns,
        class_names=['0','1'],
        filled=True,
        rounded=True,
        special_characters=True)
    graph = graphviz.Source(dot_data)
    #st.graphviz_chart(graph)

    #dt_option = st.radio("Do you want to perform Decision Tree using SMOTE:", ["Yes", "No"])
    #if st.button("Select"):
        #if dt_option == "Yes":

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

    DT1 = DecisionTreeClassifier(max_depth=10, criterion="entropy", min_samples_split=10)
    DT1_fit = DT1.fit(X_train_smote, y_train_smote)
    DT1_pred = DT1_fit.predict(X_test_smote)

    st.markdown("<h2 style='text-align: center;color: #5fb4fb;'><u>CLASSIFICATION REPORT</u></h2>", unsafe_allow_html=True)

    st.text(classification_report(y_test_smote, DT1_pred))

    st.markdown('<hr style="border-top: 1px solid; width: 75%; margin: 0 auto;">', unsafe_allow_html=True)

    st.markdown('<h3 style="color: #5fb4fb;">METRICS:</h3>', unsafe_allow_html=True)

    accuracy_dt1 = accuracy_score(y_test_smote, DT1_pred)
    precision_dt1 = precision_score(y_test_smote, DT1_pred)
    recall_dt1 = recall_score(y_test_smote, DT1_pred)

    st.write(f"<p><strong>Accuracy using SMOTE:</strong> {accuracy_dt1:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Precision using SMOTE:</strong> {precision_dt1:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Recall using SMOTE:</strong> {recall_dt1:.2f}</p>", unsafe_allow_html=True)

    import matplotlib.pyplot as plt

    feature_importances = DT1.feature_importances_
    feature_names = X.columns.tolist()

    plt.figure(figsize=(10, 8))
    plt.barh(feature_names, feature_importances)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title('Feature Importance for Decision Tree with SMOTE')
    plt.savefig('feature_importance.png')  #Save the plot as an image

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_test_smote, DT1_pred)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Decision Tree with SMOTE ROC')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')  #Save the plot as an image

    cnf_matrix = confusion_matrix(y_test_smote, DT1_pred)

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



    col1, col2 = st.columns([1, 1])

    #First Image
    with col1:
        st.image("feature_importance.png", use_column_width=True, caption="Feature Importance for Decision Tree")

    #Second Image
    with col2:
        st.image("roc_curve.png", use_column_width=True, caption="Decision Tree ROC")

    #Third Image
    with st.container():
        st.image("confusion_matrix.png", width=350, caption="Confusion Matrix")

    import plotly.express as px

    df = pd.DataFrame({'Feature Name': feature_names, 'Feature Importance': feature_importances})

    fig = px.bar(df, x='Feature Importance', y='Feature Name', orientation='h', title='Interactive Feature Importance for Decision Tree with SMOTE')
    fig.update_layout(xaxis_title='Feature Importance', yaxis_title='Feature Name')

    st.plotly_chart(fig, use_container_width=True)


    #Remove saved images from disk
    import os
    os.remove("feature_importance.png")
    os.remove("roc_curve.png")
    os.remove("confusion_matrix.png") 

    st.write(f"<p style='color:#5fb4fb'><strong>Fine tuning Decision Tree (Caution: This might take longer to run!)</strong></p>", unsafe_allow_html=True)       
        

    #dt_option = st.radio("Do you want to fine tune Decision Tree:", ["Yes", "No"])
    #if st.button("Select"):
        #if dt_option == "Yes":       
    from sklearn.model_selection import GridSearchCV
    parameters = {'max_depth': range(5, 150, 25),
        'splitter':['best'],
        'min_samples_leaf': range(50, 150, 50),
        'min_weight_fraction_leaf':[0.0],
        'class_weight':[None],
        'min_samples_split': range(50, 150, 50),
        'random_state':[2],
        'criterion': ["entropy", "gini"],
        }
    
    decision_tree_tuned = DecisionTreeClassifier()

    tree = GridSearchCV(DT, parameters, 
                        cv=5, 
                    scoring="recall")
    
    tree.fit(X_train_smote, y_train_smote)

    DT_tuned = DecisionTreeClassifier(criterion='entropy', max_depth=30, min_samples_leaf=50, min_samples_split=50, splitter='best', random_state=2)
    DT_tuned_fit = DT_tuned.fit(X_train_smote, y_train_smote)
    DT_tuned_pred = DT_tuned_fit.predict(X_test_smote)

    Train_accuracy = DT_tuned_fit.score(X_train_smote, y_train_smote)
    Test_accuracy = recall_score(y_test_smote, DT_tuned_pred)

    st.write(f"<p><strong>Training Accuracy:</strong> {Train_accuracy:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Testing Recall:</strong> {Test_accuracy:.2f}</p>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center;color: #5fb4fb;'><u>CLASSIFICATION REPORT</u></h2>", unsafe_allow_html=True)

    st.text(classification_report(y_test_smote, DT_tuned_pred))

    st.markdown('<hr style="border-top: 1px solid; width: 75%; margin: 0 auto;">', unsafe_allow_html=True)

    st.markdown('<h3 style="color: #5fb4fb;">METRICS:</h3>', unsafe_allow_html=True)

    accuracy_dt2 = accuracy_score(y_test_smote, DT_tuned_pred)
    precision_dt2 = precision_score(y_test_smote, DT_tuned_pred)
    recall_dt2 = recall_score(y_test_smote, DT_tuned_pred)

    st.write(f"<p><strong>Accuracy with fine-tuning:</strong> {accuracy_dt2:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Precision with fine-tuning:</strong> {precision_dt2:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Recall with fine-tuning:</strong> {recall_dt2:.2f}</p>", unsafe_allow_html=True)


    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    feature_importances = DT_tuned_fit.feature_importances_
    feature_names = X.columns.tolist()

    plt.figure(figsize=(10, 8))
    plt.barh(feature_names, feature_importances)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title('Feature Importance for Decision Tree with best parameters')
    plt.savefig('feature_importance.png')  #Save the plot as an image

    cnf_matrix = confusion_matrix(y_test_smote, DT_tuned_pred)

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

    fpr, tpr, _ = roc_curve(y_test_smote, DT_tuned_pred)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Decision Tree with best hyperparameters')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')  #Save the plot as an image


    col1, col2 = st.columns([1, 1])

    #First Image
    with col1:
        st.image("feature_importance.png", use_column_width=True, caption="Feature Importance for Decision Tree")

    #Second Image
    with col2:
        st.image("confusion_matrix.png", use_column_width=True, caption="Confusion Matrix")

    coll1, coll2 = st.columns([1, 1])
    with coll1:
        st.image("roc_curve.png", width=350, caption="ROC Curve")

    #with coll2:
        #st.image("roc_curve2.png", width=350, caption="ROC Curve with SMOTE")    

    #Remove saved images from disk
    import os
    os.remove("feature_importance.png")
    os.remove("confusion_matrix.png")
    os.remove("roc_curve.png")    
    #os.remove("roc_curve2.png")   


    from sklearn.tree import export_graphviz
    import graphviz

    dot_data = export_graphviz(
        DT_tuned_fit,
        out_file=None,
        feature_names=X.columns,
        class_names=['0','1'],
        filled=True,
        rounded=True,
        special_characters=True)
    graph = graphviz.Source(dot_data)
    #st.graphviz_chart(graph)


    import matplotlib.pyplot as plt

    precision, recall, accuracy = [precision_dt, precision_dt1, precision_dt2], [recall_dt, recall_dt1, recall_dt2], [accuracy_dt, accuracy_dt1, accuracy_dt2]

    labels = ['Decision Tree', 'DT - SMOTE', 'DT - fine tuned']

    X_axis = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.bar(labels, precision, width=0.25, label='Precision')
    ax.bar(X_axis+0.25, recall, width=0.25, label='Recall')
    ax.bar(X_axis+0.5, accuracy, width=0.25, label='Accuracy')

    ax.set_xlabel('Decision Tree')
    ax.set_ylabel('Score')
    ax.set_title('DT Scores')
    ax.legend()
    st.pyplot()
