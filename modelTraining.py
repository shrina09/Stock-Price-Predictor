'''
CIS4780: Computational Intelligence 
Authors: Puneet Sandher, Diya Parmar, Shrina Patel, Adina Mubbashir, Ryan Nguyen
Date: Friday, November 29, 2024
Purpose: This program trains a model using long short-term memory model
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sqlite3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load processed data from the database
def loadProcessedData(db_path="stocks_train.db"):
    try:
        print("Connected to database")
        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM stocks_train"
        data = pd.read_sql_query(query, conn)
        conn.close()

        if data.empty:
            print("ERROR: No data found in stocks_train table.")
            return None

        # LSTM Model does not accept  infinity values and it is replaced with NaN
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with NaN values
        data = data.dropna()

        if data.empty:
            print("Error: Data is empty after dropping NaN or infinity values.")
            return None

        print(f"Data successfully loaded: {data.shape[0]} rows.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Prepare the data for LSTM training
def prepareData(data):
    try:

        # Features for the model
        features = ['day_of_week', 'price_difference', 'daily_return', 'volume_return']
        # Predicts the close price of a stock
        target = 'close'  

        # Create X (features) and y (target)
        X = data[features].values
        y = data[target].values

        # LSTM does not work for every large models, and are removed. 
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Input X or y contains infinity or a value too large for dtype('float64').")

        # Since, data was removed it will be scaled 
        scalerX = MinMaxScaler(feature_range=(0, 1))
        X = scalerX.fit_transform(X)

        scalerY = MinMaxScaler(feature_range=(0, 1))
        y = scalerY.fit_transform(y.reshape(-1, 1))

        # Reshape X for LSTM 
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None, None

# Create LSTM Model
def lstmModel(input_shape):
    
    # Initalize a sequential model
    model = Sequential()
    # Add first LSTM layer with 25 units
    model.add(LSTM(25, return_sequences=True, input_shape=input_shape))

    # Add dropout layer to prevent overfitting
    model.add(Dropout(0.2))

    # Add second LSTM layer with 25 units
    model.add(LSTM(25, return_sequences=False))

    # Add dropout layer to prevent overfitting
    model.add(Dropout(0.2))
    
    model.add(Dense(1))  
    optimizer = Adam(learning_rate=0.001) 

    # Create model with MSE as loss and use MAE as a performance metric 
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
    return model

# Train the model using K-Fold cross-validation and return evaluation metrics
def modelTraining(X, y):
    kFolds = KFold(n_splits=10, shuffle=True, random_state=42)
    fold = 1

    # To store evaluation metrics (MAE and RMSE) for each fold
    metrics = []  

    # 10-fold cross validation
    for trainIndex, testIndex in kFolds.split(X):
        print(f"Training Fold Number {fold}")
        xTrain, xTest = X[trainIndex], X[testIndex]
        yTrain, yTest = y[trainIndex], y[testIndex]

        # Define the model
        model = lstmModel((xTrain.shape[1], xTrain.shape[2]))

        # Train the model
        history = model.fit(xTrain, yTrain, batch_size=32, validation_data=(xTest, yTest), verbose=1)

        # Make predictions and calculate metrics
        predictions = model.predict(xTest)
        
        # Flatten predictions to 1D
        predictions = predictions.flatten()

        # Calculate metrics
        mae = mean_absolute_error(yTest, predictions)
        rmse = np.sqrt(mean_squared_error(yTest, predictions))
        metrics.append((mae, rmse))

        print(f"Fold {fold} training complete. MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        fold += 1

    return metrics

# Evaluate the model using metrics 
def evaluateModel(metrics):
    try:
        # Calculate average MAE and RMSE
        maes = [metric[0] for metric in metrics]
        rmses = [metric[1] for metric in metrics]

        # Display evaluation metrics
        print(f"Mean Absolute Errors (MAE): {maes}")
        print(f"Root Mean Squared Errors (RMSE): {rmses}")
        print(f"Average MAE: {np.mean(maes):.4f}, Std Dev MAE: {np.std(maes):.4f}")
        print(f"Average RMSE: {np.mean(rmses):.4f}, Std Dev RMSE: {np.std(rmses):.4f}")
    except Exception as e:
        print(f"Error evaluating model: {e}")


def main():

    # Load processed data
    data = loadProcessedData()
    if data is None:
        print("Error: Data load failure")
        return

    # Adjust data for training model
    X, y = prepareData(data)
    if X is None or y is None:
        print("Error: Failed to prepare data for LSTM model.")
        return

    # Train the model with LSTM
    metrics = modelTraining(X, y)

    # Evaluate the model
    evaluateModel(metrics)

if __name__ == "__main__":
    main()

