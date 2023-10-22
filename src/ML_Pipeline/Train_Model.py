from ML_Pipeline import Utils
from tensorflow import keras
from tensorflow.keras import layers

# Function to train the machine learning model
def train(model, x_train, y_train):
    # Fit the model with the training data
    model.fit(x_train, y_train, batch_size=64, epochs=20)
    return model

# Function to initiate the model and training data
def fit(data):
    columns = data.columns

    # Prepare the training data
    x_train = data.drop(Utils.TARGET, axis=1).values  # Features
    y_train = data[Utils.TARGET].values  # Target labels

    print("Training data shape:", x_train.shape, y_train.shape)

    # Define a Sequential model using Keras
    model = keras.Sequential(
        [
            layers.InputLayer(input_shape=(x_train.shape[1])),  # Input layer with the number of features
            layers.Dense(32, activation="relu"),  # Hidden layer with 32 neurons and ReLU activation
            layers.Dense(32, activation="relu"),  # Another hidden layer
            layers.Dense(3, activation="softmax"),  # Output layer with 3 neurons for classification
        ]
    )

    # Define the optimizer and loss function
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    print("Model summary:")
    model.summary()

    # Train the model with the training data
    model = train(model, x_train, y_train)

    return model, columns
