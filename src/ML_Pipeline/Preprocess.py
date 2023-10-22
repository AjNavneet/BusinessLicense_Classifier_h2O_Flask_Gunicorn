import numpy as np
import pandas as pd

# Function to cleanup data, convert to numerical variables and impute wherever required
def cleanup(data):
    # Match legal business names in uppercase
    data['LEGAL_BUSINESS_NAME_MATCH'] = data \
        .apply(lambda x: 1 if str(x['LEGAL_NAME'].upper()) in str(x['DOING_BUSINESS_AS_NAME']).upper()
                              or str(x['DOING_BUSINESS_AS_NAME']).upper() in str(x['LEGAL_NAME']).upper() else 0,
               axis=1)

    # Replace and simplify license descriptions
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Motor Vehicle Repair : Engine Only (Class II)',
                                                                      'Motor Vehicle Repair')
    # (Other replacements for license descriptions)

    # Remove dots from legal business names
    data['LEGAL_NAME'] = data['LEGAL_NAME'].str.replace('.', '', regex=False)
    data['DOING_BUSINESS_AS_NAME'] = data['DOING_BUSINESS_AS_NAME'].str replace('.', '', regex=False)

    # Impute business types based on keywords in legal names and DBAs
    data['BUSINESS_TYPE'] = 'PVT'
    # (Imputations based on keywords)

    # Impute missing ZIP codes and create a ZIP_CODE_MISSING flag
    data['ZIP_CODE'].fillna(-1, inplace=True)
    data['ZIP_CODE_MISSING'] = data.apply(lambda x: 1 if x['ZIP_CODE'] == -1 else 0, axis=1)

    # Impute SSAs
    data['SSA'].fillna(-1, inplace=True)

    # Impute and convert APPLICATION_REQUIREMENTS_COMPLETE to binary
    data['APPLICATION_REQUIREMENTS_COMPLETE'].fillna(-1, inplace=True)
    data['APPLICATION_REQUIREMENTS_COMPLETE'] = data\
        .apply(lambda x: 0 if x['APPLICATION_REQUIREMENTS_COMPLETE'] == -1 else 1, axis=1)

    return data

# Function to encode categorical variables
def categorical_encode(data):
    # Get the list of predictors and encode categorical variables
    try:
        from ML_Pipeline.Utils import PREDICTORS
        final_df = data[PREDICTORS + ["LICENSE_STATUS"]]
        final_df = pd.get_dummies(final_df, columns=['APPLICATION_TYPE', 'CONDITIONAL_APPROVAL', 'LICENSE_CODE',
                                                     'LICENSE_DESCRIPTION', 'BUSINESS_TYPE', 'LICENSE_STATUS'])
    except:
        # For test data
        final_df = data[PREDICTORS]
        final_df = pd.get dummies(final_df, columns=['APPLICATION_TYPE', 'CONDITIONAL_APPROVAL', 'LICENSE_CODE',
                                                     'LICENSE_DESCRIPTION', 'BUSINESS_TYPE'])

    return final_df

# Function to call dependent functions
def apply(data):
    print("Preprocessing started....")

    data = cleanup(data)
    print("Data cleanup completed....")

    data = categorical_encode(data)
    print("Categorical encoding completed....")

    data = data.loc[:, ~data.columns.duplicated()]

    print("Preprocessing completed....")
    return data
