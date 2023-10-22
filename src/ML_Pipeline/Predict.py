import pandas as pd

# Define a function to make predictions
def init(test_data, model, columns):
    # Preprocess test data to do categorical encoding and match columns expected by the model

    # Find columns that are in the model's columns but not in the test data
    new_cols = [x for x in columns if x not in test_data.columns]

    # Create a new DataFrame with missing columns and fill missing values with 0
    new_df = pd.DataFrame(columns=new_cols, index=range(test_data.shape[0])
    new_df.fillna(0, inplace=True)

    # Concatenate the new DataFrame with the test data
    test_data = pd.concat([test_data, new_df.reindex(test_data.index)], axis=1)

    # Reorder columns to match the expected order
    test_data = test_data[columns]

    # Remove any duplicated columns
    test_data = test_data.loc[:, ~test_data.columns.duplicated()]

    # Remove target columns if they exist
    from ML_Pipeline import Utils
    for col in Utils.TARGET:
        try:
            test_data = test_data.drop(col, axis=1)
        except:
            continue

    # Get the values from the preprocessed test data
    x_test = test_data.values

    # Make predictions using the pre-trained machine learning model
    predict = model.predict(x_test)

    return predict
