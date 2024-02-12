***********************************Diabetes Readmission Predictor***********************************

Running Web-application in local requirements:
To use the web-app, you'll need be needing Python 3.8 or higher and the following libraries: streamlit, pandas, numpy, matplotlib, scikit-learn (with modules: preprocessing, ensemble, impute, metrics, model_selection, linear_model), imblearn, xgboost, and seaborn.

* streamlit - Used for creating data web applications
* pandas - Used for data manipulation and analysis
* numpy - Used for various numerical operations
* matplotlib - Used for creating static, and interactive visualizations with data
* sklearn.preprocessing - Used for StandardScaler and LabelEncoder in data preprocessing
* sklearn.impute - Used for KNNImputer in handling missing data
* sklearn.metrics - Used for evaluating the model performance
* sklearn.model_selection - Used for train_test_split in splitting the data into training and test sets.
* sklearn.linear_model - Used in classification tasks
* sklearn.ensemble - Used for RandomForestClassifier, AdaBoostClassifier
* sklearn.tree - Used for DecisionTreeClassifier 
* sklearn.naive_bayes - Used for Naive Bayes model
* skleanrn.model_selection - Used for GridSearch in fine-tuning
* seaborn - Used for statistical data visualization
* xgboost - Used for XGBoost model
* imblearn - Used for performing SMOTE


Pip Installations:
* pip install streamlit
* pip install pandas
* pip install numpy
* pip install matplotlib
* pip install scikit-learn
* pip install seaborn
* pip install statsmodels
* pip install -U kaleido
* pip install plotly
* pip install xgboost
* pip install imblearn

We have covered all-most every required installation in the above list. However, there might be additional installations required specific to the user’s system/environment  if any issues are encountered.

Running the application:

Step 1: Performing the required installations, and download the project contents into a directory.

Step 2: Using the terminal, navigate to the project directory where the streamlit script (app.py) is located. This can be done using the cd command.
Example: cd /Users/dhaya/OneDrive/Desktop//DIC/Phase3/src

Step 3: Once reaching the project directory, the web-app can be launched with the following command: streamlit run app.py
If the browser did not automatically open, go to the URL where the web-app is running, usually http://localhost:8501. This can be done by either clicking on the URL in the terminal to open it in the web browser, or manually opening the preferred web browser and then entering the URL.

Information about code structure:

Inside the src folder, there will be various files and folders which will be necessary for correct running of the application. Inside the asset folder, images which we used are present. Inside the models folder, the trained and saved models will be present. We have made the application modular by having separate code files for various functions and importing them when necessary. The app.py is the main python script which import other necessary models and functions present in other files like clean_auto.py (which performs automatic cleaning), and adaBoost.py, naiveBayes.py etc., (functions for training and visualizing based on the chosen model).


