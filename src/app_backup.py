import streamlit as st
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sb
import seaborn as sns
from sklearn import metrics

def clean_auto(df):
    #Replacing unknown (?) values with NaN
    df = df.replace('?', np.nan)

    #Removing duplicate records
    df.drop_duplicates(inplace=True)

    #Dropping columns with high percentage of missing values (> 40%)
    df = df.drop(['weight', 'payer_code', 'medical_specialty'], axis=1)

    #Dropping additional column with unique identifer which will not be useful predicting readmission
    df = df.drop(['encounter_id', 'patient_nbr'], axis=1)

    df['readmitted'] = df['readmitted'].replace({'NO': 0, '>30': 0, '<30': 1})

    def check_diabetes(diag):
        if str(diag).startswith('250'):
            return 1
        else:
            return 0

    # Impute missing values - categorical -> mode values

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

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    return X, y

def logisticRegression(X, y):
    st.write("Entered logisticRegression")
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

     #perform SMOTE on dataset
    from imblearn.over_sampling import SMOTE
    smt = SMOTE(random_state=20)
    X_resampled, Y_resampled = smt.fit_resample(X, y)
    X_resampled = pd.DataFrame(X_resampled, columns = list(X.columns))
    X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_resampled, Y_resampled, test_size=0.20, random_state=20)  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20, shuffle=True)
    LR = LogisticRegression()
    LR.fit(X_train, y_train)
    LR_pred = LR.predict(X_test)

    classification_report_str = classification_report(y_test, LR_pred)
    accuracy_lr = accuracy_score(y_test, LR_pred)
    precision_lr = precision_score(y_test, LR_pred)
    recall_lr = recall_score(y_test, LR_pred)

    st.write("Classification Report:")
    st.text(classification_report_str)

    st.write(f"Accuracy: {accuracy_lr:.2f}")
    st.write(f"Precision: {precision_lr:.2f}")
    st.write(f"Recall: {recall_lr:.2f}")

    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_test, LR_pred)

    plt.plot(recall, precision, label='Precision-recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-recall curve')
    plt.legend(loc="lower left")

    st.pyplot(plt)

    fpr, tpr, _ = metrics.roc_curve(y_test, LR_pred)
    auc = metrics.roc_auc_score(y_test, LR_pred)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)

    st.pyplot(plt)

    cnf_matrix = metrics.confusion_matrix(y_test, LR_pred)

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

    st.pyplot(fig)

    lr_option = st.radio("Do you want to perform LR using weight balanced:", ["Yes", "No"])
    if st.button("Select"):
       if lr_option == "Yes": 
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

            st.write("Classification Report for LR using weight balanced:")
            st.text(classification_report_str1)

            st.write(f"Accuracy for LR using weight balanced: {accuracy_lr1:.2f}")
            st.write(f"Precision for LR using weight balanced: {precision_lr1:.2f}")
            st.write(f"Recall for LR using weight balanced: {recall_lr1:.2f}")

            fpr, tpr, _ = metrics.roc_curve(y_test, LR1_pred)
            auc = metrics.roc_auc_score(y_test, LR1_pred)

            #plot ROC curve
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label="data 1, auc=" + str(auc))
            ax.legend(loc=4)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            st.pyplot(fig)

            from sklearn.metrics import precision_recall_curve
            import matplotlib.pyplot as plt

            #compute Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_test, LR1_pred)

            #plot Precision-Recall curve
            fig, ax = plt.subplots()
            ax.plot(recall, precision, label='Precision-recall curve')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-recall curve')
            ax.legend(loc='lower left')
            st.pyplot(fig)

            cnf_matrix = metrics.confusion_matrix(y_test, LR1_pred)
            class_names = [0, 1]

            #plot confusion matrix
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
            st.pyplot(fig)

       else:
           return
       
    lr_option2 = st.radio("Do you want to perform LR using SMOTE:", ["Yes", "No"])
    if st.button("Select"):
        if lr_option2 == "Yes":
            LR2 = LogisticRegression(solver='sag',class_weight='balanced')
            LR2_fit = LR2.fit(X_train_smote, y_train_smote)
            LR2_pred_smote = LR2_fit.predict(X_test_smote)

            classification_rep = classification_report(y_test_smote, LR2_pred_smote)
            accuracy_lr2 = accuracy_score(y_test_smote, LR2_pred_smote)
            precision_lr2 = precision_score(y_test_smote, LR2_pred_smote)
            recall_lr2 = recall_score(y_test_smote, LR2_pred_smote)

            st.write("Classification Report for LR using SMOTE")
            st.text(classification_rep)

            st.write("Accuracy for LR using SMOTE: ", accuracy_lr2)
            st.write("Precision for LR using SMOTE: ", precision_lr2)
            st.write("Recall for LR using SMOTE: ", recall_lr2)

            from sklearn.metrics import precision_recall_curve
            import matplotlib.pyplot as plt

            precision, recall, _ = precision_recall_curve(y_test_smote, LR2_pred_smote)

            fig, ax = plt.subplots()
            ax.plot(recall, precision, label='Precision-recall curve')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-recall curve')
            ax.legend(loc='lower left')
            st.pyplot(fig)

            from matplotlib import pyplot as plt
            from sklearn import metrics

            #compute ROC curve
            fpr, tpr, _ = metrics.roc_curve(y_test_smote, LR2_pred_smote)
            auc = metrics.roc_auc_score(y_test_smote, LR2_pred_smote)

            #plot ROC curve
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label="auc" + str(auc))
            ax.legend(loc=4)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            st.pyplot(fig)

            cnf_matrix = metrics.confusion_matrix(y_test_smote, LR2_pred_smote)
            class_names = [0, 1]

            #plot confusion matrix
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
            st.pyplot(fig)

            import matplotlib.pyplot as plt

            precision, recall, accuracy = [precision_lr, precision_lr1, precision_lr2], [recall_lr, recall_lr1, recall_lr2], [accuracy_lr, accuracy_lr1, accuracy_lr2]
            labels = ['Logistic Regression', 'LR - weight balanced', 'LR - SMOTE - WB']

            X_axis = np.arange(len(labels))
            fig, ax = plt.subplots()
            ax.bar(labels, precision, width=0.25, label='Precision')
            ax.bar(X_axis+0.25, recall, width=0.25, label='Recall')
            ax.bar(X_axis+0.5, accuracy, width=0.25, label='Accuracy')

            ax.set_xlabel('Logistic Regression')
            ax.set_ylabel('Score')
            ax.set_title('LR Scores')
            ax.legend()

            st.pyplot(fig)
        else:
            return
           

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

    st.write("Classification Report:")
    st.write(classification_report(y_test, DT_pred))

    accuracy_dt = accuracy_score(y_test, DT_pred)
    precision_dt = precision_score(y_test, DT_pred)
    recall_dt = recall_score(y_test, DT_pred)
    st.write("Accuracy:", accuracy_dt)
    st.write("Precision:", precision_dt)
    st.write("Recall:", recall_dt)

    import matplotlib.pyplot as plt

    feature_importances = DT.feature_importances_
    feature_names = X.columns.tolist()

    plt.figure(figsize=(10, 8))
    plt.barh(feature_names, feature_importances)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title('Feature Importance for Decision Tree')
    st.pyplot()

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_test, DT_pred)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Decision Tree initial ROC')
    plt.legend(loc="lower right")
    st.pyplot()

    from sklearn import metrics

    cnf_matrix = metrics.confusion_matrix(y_test, DT_pred)

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
    st.pyplot()


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
    st.graphviz_chart(graph)

    dt_option = st.radio("Do you want to perform Decision Tree using SMOTE:", ["Yes", "No"])
    if st.button("Select"):
        if dt_option == "Yes":
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

            DT1 = DecisionTreeClassifier(max_depth=10, criterion="entropy", min_samples_split=10)
            DT1_fit = DT1.fit(X_train_smote, y_train_smote)
            DT1_pred = DT1_fit.predict(X_test_smote)

            st.write("Classification Report:")
            st.write(classification_report(y_test_smote, DT1_pred))

            accuracy_dt1 = accuracy_score(y_test_smote, DT1_pred)
            precision_dt1 = precision_score(y_test_smote, DT1_pred)
            recall_dt1 = recall_score(y_test_smote, DT1_pred)

            st.write("Accuracy:", accuracy_dt1)
            st.write("Precision:", precision_dt1)
            st.write("Recall:", recall_dt1)

            import matplotlib.pyplot as plt

            feature_importances = DT1.feature_importances_
            feature_names = X.columns.tolist()

            plt.figure(figsize=(10, 8))
            plt.barh(feature_names, feature_importances)
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature Name')
            plt.title('Feature Importance for Decision Tree with SMOTE')
            st.pyplot()


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
            st.pyplot()


            cnf_matrix = metrics.confusion_matrix(y_test_smote, DT1_pred)

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
            st.pyplot()

        else:
            return
        

    dt_option = st.radio("Do you want to fine tune Decision Tree:", ["Yes", "No"])
    if st.button("Select"):
        if dt_option == "Yes":        
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

            st.write('Training Accuracy:', Train_accuracy)
            st.write('Testing Recall:', Test_accuracy)
            st.write('Classification Report:')
            st.write(classification_report(y_test_smote, DT_tuned_pred))

            accuracy_dt2 = accuracy_score(y_test_smote, DT_tuned_pred)
            precision_dt2 = precision_score(y_test_smote, DT_tuned_pred)
            recall_dt2 = recall_score(y_test_smote, DT_tuned_pred)

            st.write("Accuracy:", accuracy_dt2)
            st.write("Precision:", precision_dt2)
            st.write("Recall:", recall_dt2)


            import matplotlib.pyplot as plt

            feature_importances = DT_tuned_fit.feature_importances_
            feature_names = X.columns.tolist()

            plt.figure(figsize=(10, 8))
            plt.barh(feature_names, feature_importances)
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature Name')
            plt.title('Feature Importance for Decision Tree with best parameters')
            st.pyplot()

            cnf_matrix = metrics.confusion_matrix(y_test_smote, DT_tuned_pred)

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
            st.pyplot()


            from sklearn.metrics import roc_curve, auc
            import matplotlib.pyplot as plt

            fpr, tpr, _ = roc_curve(y_test_smote, DT_tuned_pred)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Decision Tree with best hyperparameters')
            plt.legend(loc="lower right")
            st.pyplot()

            fpr, tpr, _ = metrics.roc_curve(y_test_smote, DT_tuned_pred)
            auc = metrics.roc_auc_score(y_test_smote, DT_tuned_pred)

            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Decision Tree with best hyperparameters')
            plt.legend(loc="lower right")
            st.pyplot()


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
            st.graphviz_chart(graph)


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

        else:
            return

def randomForestClassifier():

     #perform SMOTE on dataset
    from imblearn.over_sampling import SMOTE
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

    st.write(classification_report(y_test, rm_pred))
    accuracy_rm = accuracy_score(y_test, rm_pred)
    precision_rm = precision_score(y_test, rm_pred)
    recall_rm = recall_score(y_test, rm_pred)

    st.write('Accuracy:', accuracy_score(y_test, rm_pred))
    st.write('Precision:', precision_score(y_test, rm_pred))
    st.write('Recall:', recall_score(y_test, rm_pred))

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
    st.pyplot()

    from sklearn import metrics
    cnf_matrix = metrics.confusion_matrix(y_test, rm_pred)

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
    st.pyplot()

    fpr, tpr, thresholds = roc_curve(y_test, rm_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Random Forest')
    plt.legend(loc="lower right")
    st.pyplot()

    rf_option = st.radio("Do you want to perfrom Random Forest with SMOTE:", ["Yes", "No"])           
    if st.button("Select"):
        if rf_option == "Yes":
            rm_smote = RandomForestClassifier(n_estimators=10, max_depth=25, criterion="gini", min_samples_split=10)
            rm_smote_fit = rm_smote.fit(X_train_smote, y_train_smote)
            rm_smote_pred = rm_smote_fit.predict(X_test_smote)

            st.write('Classification Report for Random Forest with SMOTE:')
            st.text(classification_report(y_test_smote, rm_smote_pred))
            accuracy_rm1 = accuracy_score(y_test_smote, rm_smote_pred)
            precision_rm1 = precision_score(y_test_smote, rm_smote_pred)
            recall_rm1 = recall_score(y_test_smote, rm_smote_pred)

            st.write('Evaluation Metrics for Random Forest with SMOTE:')
            st.write('Accuracy:', accuracy_rm1)
            st.write('Precision:', precision_rm1)
            st.write('Recall:', recall_rm1)

            from sklearn.metrics import confusion_matrix

            cnf_matrix = confusion_matrix(y_test_smote, rm_smote_pred)

            st.write('Confusion Matrix for Random Forest with SMOTE:')
            fig, ax = plt.subplots()
            tick_marks = np.arange(len(['0', '1']))
            plt.xticks(tick_marks, ['0', '1'])
            plt.yticks(tick_marks, ['0', '1'])
            sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g', cbar=False, square=True)
            plt.xlabel('Predicted label')
            plt.ylabel('Actual label')
            st.pyplot(fig)

            from sklearn.metrics import roc_curve, auc

            fpr, tpr, thresholds = roc_curve(y_test_smote, rm_smote_pred)
            roc_auc = auc(fpr, tpr)

            st.pyplot(plt.figure(figsize=(10, 8)))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for Random Forest with SMOTE')
            plt.legend(loc="lower right")
            st.pyplot()

        else:
            return
        
    rf_option = st.radio("Do you want to perfrom Random Forest with best parameter estimation:", ["Yes", "No"])           
    if st.button("Select"):
        if rf_option == "Yes":
            from sklearn.model_selection import GridSearchCV
            parameter={'n_estimators':np.arange(1,80)}
            gs = GridSearchCV(rm,parameter,cv=3)
            gs.fit(X_train,y_train)

            st.write('Best Hyperparameters:', gs.best_params_)

            rm_finetuned = RandomForestClassifier(n_estimators=66)
            rm_finetuned_fit = rm_finetuned.fit(X_train_smote,y_train_smote) 
            rm_finetuned_pred = rm_finetuned_fit.predict(X_test_smote)
            rm_finetuned_training_accuracy = rm_finetuned_fit.score(X_train_smote,y_train_smote)
            rm_finetuned_testing_accuracy = accuracy_score(y_test_smote,rm_finetuned_pred)

            st.write('Training Accuracy :',rm_finetuned_training_accuracy)
            st.write('Testing Accuracy  :',rm_finetuned_testing_accuracy)
            st.write(classification_report(y_test_smote,rm_finetuned_pred))

            accuracy_rm2 = accuracy_score(y_test_smote, rm_finetuned_pred)
            precision_rm2 = precision_score(y_test_smote, rm_finetuned_pred)
            recall_rm2 = recall_score(y_test_smote, rm_finetuned_pred)

            st.write("Random Forest - Accuracy:", accuracy_rm2)

            st.write("Random Forest - Precision:", precision_rm2)

            st.write("Random Forest - Recall:", recall_rm2)

            import matplotlib.pyplot as plt

            feature_importances = rm_finetuned.feature_importances_
            feature_names = X.columns.tolist()

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(feature_names, feature_importances)
            ax.set_xlabel('Feature Importance')
            ax.set_ylabel('Feature Name')
            ax.set_title('Feature Importance for Random Forest')
            st.pyplot(fig)

            from sklearn.metrics import roc_curve, auc

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

            st.pyplot()


            cnf_matrix = metrics.confusion_matrix(y_test_smote, rm_finetuned_pred)

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

            st.pyplot(fig)


            fpr, tpr, _ = metrics.roc_curve(y_test_smote, rm_finetuned_pred)
            auc = metrics.roc_auc_score(y_test_smote, rm_finetuned_pred)

            fig, ax = plt.subplots()
            plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
            plt.legend(loc=4)
            plt.title('ROC Curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')

            st.pyplot(fig)


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
        else:
            return
            



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

    st.write("Training accuracy: ", adb_train_accuracy)
    st.write("Testing accuracy: ", adb_test_accuracy)

    classification = classification_report(y_test, adb_pred)
    st.write("Classification Report:\n", classification)


    accuracy_adb = accuracy_score(y_test, adb_pred)
    precision_adb = precision_score(y_test, adb_pred)
    recall_adb = recall_score(y_test, adb_pred)

    st.write("Accuracy: ", accuracy_adb)
    st.write("Precision: ", precision_adb)
    st.write("Recall: ", recall_adb)

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

    st.pyplot(fig)

    from sklearn import metrics

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

    st.pyplot(fig)

    rf_option = st.radio("Do you want to perfrom AdaBoost with hyperparameter tuning:", ["Yes", "No"])           
    if st.button("Select"):
        if rf_option == "Yes":
            from sklearn.ensemble import AdaBoostClassifier
            from sklearn.metrics import classification_report

            adb_tuned = AdaBoostClassifier(random_state=20, n_estimators=73, learning_rate=0.01)

            adb_tuned.fit(X_train_smote, y_train_smote)
            adb_tuned_pred = adb_tuned.predict(X_test_smote)
            adb_train_accuracy_tuned = adb_tuned.score(X_train_smote, y_train_smote)
            adb_test_accuracy_tuned = adb_tuned.score(X_test_smote, y_test_smote)

            st.write("Training accuracy: ", adb_train_accuracy_tuned)
            st.write("Testing accuracy: ", adb_test_accuracy_tuned)

            classification_tuned = classification_report(y_test_smote, adb_tuned_pred)
            st.write("Classification Report:\n", classification_tuned)

            accuracy_adb1 = accuracy_score(y_test_smote, adb_tuned_pred)
            precision_adb1 = precision_score(y_test_smote, adb_tuned_pred)
            recall_adb1 = recall_score(y_test_smote, adb_tuned_pred)

            st.write("Accuracy: ", accuracy_adb1)
            st.write("Precision: ", precision_adb1)
            st.write("Recall: ", recall_adb1)

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

            st.pyplot(fig)


            from sklearn.model_selection import learning_curve
            import matplotlib.pyplot as plt

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

            st.pyplot(fig)

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

            st.pyplot(fig)

        else:
            return


def XGBoost(X, y):

         #perform SMOTE on dataset
    from imblearn.over_sampling import SMOTE
    smt = SMOTE(random_state=20)
    X_resampled, Y_resampled = smt.fit_resample(X, y)
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

    print('Training Accuracy is:', Train_Score_xgb)
    print('Testing Accuracy is:', Test_Score_xgb)
    print(classification_report(y_test, y_pred_xgb))

    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_test, y_pred_xgb)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, label='Precision-recall curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-recall curve')
    ax.legend(loc="lower left")

    st.pyplot(fig)

    from sklearn import metrics

    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_xgb)
    auc = metrics.roc_auc_score(y_test, y_pred_xgb)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="XG boost curve, auc="+str(auc))
    ax.legend(loc=4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')

    st.pyplot(fig)

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

    st.pyplot(fig)

    from sklearn.metrics import classification_report
    import plotly.figure_factory as ff

    report = classification_report(y_test, y_pred_xgb, output_dict=True)
    fig = ff.create_annotated_heatmap(z=[[report['0']['precision'], report['0']['recall'], report['0']['f1-score']],
                                        [report['1']['precision'], report['1']['recall'], report['1']['f1-score']]],
                                    x=['precision', 'recall', 'f1-score'],
                                    y=['Negative', 'Positive'], colorscale='Viridis')
    fig.update_layout(title='Classification Report Heat Map')

    st.plotly_chart(fig)


    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_test, y_pred_xgb, n_bins=10)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    ax.plot(prob_pred, prob_true, marker='.', label='XGBoost')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('True probability')
    ax.set_title('Calibration plot')
    ax.legend()

    st.pyplot(fig)


    xg_option = st.radio("Do you want to perfrom XGBoost with hyperparameter tuning:", ["Yes", "No"])           
    if st.button("Select"):
        if xg_option == "Yes":
            import xgboost as xgb
            from sklearn.metrics import accuracy_score, classification_report

            xgb_classifier = xgb.XGBClassifier()

            xgb_classifier.fit(X_train_smote, y_train_smote)

            y_preds_xgb = xgb_classifier.predict(X_test_smote)

            Trains_Score_xgb = xgb_classifier.score(X_train_smote, y_train_smote)
            Tests_Score_xgb = accuracy_score(y_test_smote, y_preds_xgb)

            st.write('Training Accuracy is:', Trains_Score_xgb)
            st.write('Testing Accuracy is:', Tests_Score_xgb)
            st.write(classification_report(y_test_smote, y_preds_xgb))

            from sklearn.metrics import precision_recall_curve
            import matplotlib.pyplot as plt

            precision, recall, _ = precision_recall_curve(y_test_smote, y_preds_xgb)

            plt.plot(recall, precision, label='Precision-recall curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-recall curve')
            plt.legend(loc="lower left")

            st.pyplot(plt.gcf())

            from sklearn import metrics
            import matplotlib.pyplot as plt

            fpr, tpr, _ = metrics.roc_curve(y_test_smote,  y_preds_xgb)
            auc = metrics.roc_auc_score(y_test_smote, y_preds_xgb)
            plt.plot(fpr,tpr,label="XG boost SMOTE, auc="+str(auc))
            plt.legend(loc=4)

            st.pyplot(plt.gcf())

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

            st.pyplot(plt.gcf())

            import plotly.figure_factory as ff
            from sklearn.metrics import classification_report

            report = classification_report(y_test_smote, y_preds_xgb, output_dict=True)
            fig = ff.create_annotated_heatmap(z=[[report['0']['precision'], report['0']['recall'], report['0']['f1-score']],
                                                [report['1']['precision'], report['1']['recall'], report['1']['f1-score']]],
                                            x=['precision', 'recall', 'f1-score'],
                                            y=['Negative', 'Positive'], colorscale='Viridis')
            fig.update_layout(title='Classification Report Heat Map')
            st.plotly_chart(fig, use_container_width=True)

            from sklearn.calibration import calibration_curve

            prob_true, prob_pred = calibration_curve(y_test_smote, y_preds_xgb, n_bins=10)
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
            ax.plot(prob_pred, prob_true, marker='.', label='XGBoost')
            ax.set_xlabel('Predicted probability')
            ax.set_ylabel('True probability')
            ax.set_title('Calibration plot')
            ax.legend()
            st.pyplot(fig)

        else:
            return    




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

    st.write('Training Accuracy :', NB_training_accuracy)
    st.write('Testing Accuracy  :', NB_testing_accuracy)
    st.write(classification_report(y_test,NB_pred))

    accuracy_nb = accuracy_score(y_test, NB_pred)
    precision_nb = precision_score(y_test, NB_pred)
    recall_nb = recall_score(y_test, NB_pred)

    st.write("Accuracy: ", accuracy_nb)
    st.write("Precision: ", precision_nb)
    st.write("Recall: ", recall_nb)

    from sklearn.metrics import confusion_matrix

    cnf_matrix = confusion_matrix(y_test, NB_pred)

    sns.heatmap(cnf_matrix, cmap='Blues', annot=True, fmt='g')

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    st.pyplot()

    nb_option = st.radio("Do you want to perfrom Naive Bayes with SMOTE:", ["Yes", "No"])           
    if st.button("Select"):
        if nb_option == "Yes":

            NB1=BernoulliNB()
            NB1.fit(X_train_smote,y_train_smote)
            NB_pred1=NB1.predict(X_test_smote)
            NB_training_accuracy = NB1.score(X_train_smote,y_train_smote)
            NB_testing_accuracy = accuracy_score(y_test_smote,NB_pred1)

            st.write('Training Accuracy :',NB_training_accuracy)
            st.write('Testing Accuracy  :',NB_testing_accuracy)
            st.write(classification_report(y_test_smote,NB_pred1))

            accuracy_nb1 = accuracy_score(y_test_smote,NB_pred1)
            precision_nb1 = precision_score(y_test_smote,NB_pred1)
            recall_nb1 = recall_score(y_test_smote,NB_pred1)

            st.write('Smote Accuracy:',accuracy_nb1)
            st.write('Smote Precision:',precision_nb1)
            st.write('Smote Recall:',recall_nb1)

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

            st.pyplot(fig)


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

            st.pyplot(fig)
       
        else:
            return



st.title("Diabetes Readmission Predictor")
st.write("Upload your dataset here!")

dataset = st.file_uploader("Choose your dataset - CSV file", type="csv")

X = None
y = None

if dataset is not None:
    try:
        data = pd.read_csv(dataset)
        st.write("Displaying first 5 rows of the dataset:")
        st.write(data.head())

        clean_option = st.radio("Select data cleaning option", ["Clean Automatically", "Clean Manually"])
        if st.button("Clean!"):
            if clean_option == "Clean Automatically":
                X, y = clean_auto(data)
                st.write("Automatic cleaning successfull!")

        model_option = st.radio("Select the model to train your dataset with:", ["Logistic Regression", "Decision Tree", "Random Forest Classifier", "Ada Boost", "XGBoost", "Naive Bayes"])
        
        if st.button("Train Model!"):
            #with st.spinner(f"Training with {model_option}..."):    
            st.write("Clicked Train Model!")
            if model_option == "Logistic Regression":
                st.write("Training with Logistic Regression!")
                logisticRegression(X, y)
            elif model_option == "Decision Tree":
                st.write("Training with Decision Tree!")
                decisionTree(X, y)
            elif model_option == "Random Forest Classifier":
                st.write("Training with Random Forest Classifier!")
                randomForestClassifier(X, y)
            elif model_option == "Ada Boost":
                st.write("Training with Ada Boost!")
                adaBoost(X, y)
            elif model_option == "XGBoost":
                st.write("Training with XGBoost!")
                XGBoost(X, y)
            elif model_option == "Naive Bayes":
                st.write("Training with Naive Bayes!")
                naiveBayes(X, y)


    except Exception as e:
        st.write(f"Error in reading the CSV file: {e}")
