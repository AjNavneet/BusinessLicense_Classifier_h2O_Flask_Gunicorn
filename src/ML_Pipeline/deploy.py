# Import necessary libraries and modules
import pandas as pd
from flask import Flask, request
import json
import Preprocess  # Import data preprocessing functions
import Predict  
import Utils  

# Create a Flask application
app = Flask(__name)

# Set the path to the pre-trained machine learning model and load it
model_path = '../output/dnn-model'
ml_model, columns = Utils.load_model(model_path)

# Define an endpoint for receiving POST requests
@app.post("/get_license_status")
async def get_license_status():
    # Receive data in JSON format from the POST request
    items = json.loads(request.data)

    # Create a Pandas DataFrame from the received data
    test_df = pd.DataFrame([items], columns=items.keys())

    # Apply data preprocessing to the input data
    processed_df = Preprocess.apply(test_df)

    # Get predictions from the machine learning model
    prediction = list(Predict.init(processed_df, ml_model, columns)[0])

    # Determine the class with the highest prediction probability
    max_value = max(prediction)
    max_index = prediction.index(max_value)

    # Map the class index to a human-readable status label using utility functions
    output = {"status": Utils.TARGET[max_index]}
    print(output)

    # Return the predicted license status as a JSON response
    return output

# Run the Flask application on host 0.0.0.0 and port 5000
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
