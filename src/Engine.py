# Import necessary modules and classes
from ML_Pipeline import Predict, Train_Model
from ML_Pipeline.Preprocess import apply
from ML_Pipeline.Utils import load_model, save_model
import pandas as pd
import subprocess

# Prompt the user for the task to perform (Training, Prediction, or Deployment)
val = int(input("Train - 0\nPredict - 1\nDeploy - 2\nEnter your value: "))

# Training a Machine Learning Model
if val == 0:
    # Load data from a CSV file and perform data preprocessing
    data = pd.read_csv("../input/License_Data.csv", low_memory=False).drop_duplicates().reset_index(drop=True)
    print("Data loaded into pandas dataframe")
    processed_df = apply(data)

    # Train a machine learning model and save it
    ml_model, columns = Train_Model.fit(processed_df)
    model_path = save_model(ml_model, columns)
    print("Model saved in:", model_path)

# Making Predictions with a Trained Model
elif val == 1:
    # Load a pre-trained machine learning model from a specified location
    model_path = "../output/dnn-model"
    ml_model, columns = load_model(model_path)

    # Load test data, preprocess it, and make predictions
    test_data = pd.read_csv("../input/test_data.csv", low_memory=False).drop_duplicates().reset_index(drop=True)
    processed_df = apply(test_data)
    prediction = Predict.init(processed_df, ml_model, columns)
    print(prediction)

# Deployment (Note: The code for deployment is commented out)
else:
    # For production deployment
    '''process = subprocess.Popen(['sh', 'ML_Pipeline/wsgi.sh'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True
                               )'''

    # For development deployment
    process = subprocess.Popen(['python', 'ML_Pipeline/deploy.py'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True
                               )

    for stdout_line in process.stdout:
        print(stdout_line)

    stdout, stderr = process.communicate()
    print(stdout, stderr)
