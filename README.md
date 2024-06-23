***********************************Diabetes Readmission Predictor***********************************

Screenshots:
![dataFlow1](https://github.com/dhayanesh/dataFlow/assets/63561465/da54da4d-70ad-4622-a32e-77d4c9153295)
![dataFlow2](https://github.com/dhayanesh/dataFlow/assets/63561465/eaa3450d-d3ba-4177-bdac-a58623d24a93)
![dataFlow3](https://github.com/dhayanesh/dataFlow/assets/63561465/7206aff9-c1ff-4af0-ac9e-7547899222a1)

Running Web-application in local requirements:
To use the web-app, you'll need be needing Python 3.8 or higher and the following libraries: streamlit, pandas, numpy, matplotlib, scikit-learn (with modules: preprocessing, ensemble, impute, metrics, model_selection, linear_model), imblearn, xgboost, and seaborn.

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


