import keras
import pickle

# Define the predictors and target columns
PREDICTORS = ['APPLICATION_TYPE', 'CONDITIONAL_APPROVAL', 'LICENSE_CODE', 'SSA', 'LEGAL_BUSINESS_NAME_MATCH',
              'ZIP_CODE_MISSING', 'SSA', 'APPLICATION_REQUIREMENTS_COMPLETE', 'LICENSE_DESCRIPTION', 'BUSINESS_TYPE']

TARGET = ["LICENSE_STATUS_AAC", "LICENSE_STATUS_AAI", "LICENSE_STATUS_REV"]

# Function to save the machine learning model and columns mapping
def save_model(model, columns):
    # Save the Keras model to the specified path
    model.save("../output/dnn-model")

    # Create a file to save the columns mapping using pickle
    file = open("../output/columns.mapping", "wb")
    pickle.dump(columns, file)
    file.close()

    return True

# Function to load the machine learning model and columns mapping
def load_model(model_path):
    model = None
    try:
        # Attempt to load the Keras model from the provided model_path
        model = keras.models.load_model(model_path)
    except:
        print("Please enter the correct model path")
        exit(0)

    # Load the columns mapping from the stored file using pickle
    file = open("../output/columns.mapping", "rb")
    columns = pickle.load(file)
    file.close()

    return model, columns
