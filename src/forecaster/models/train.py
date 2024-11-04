import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from forecaster.models.evaluate import calculate_metrics


def ts_prediction(X_train, y_train, X_test, y_test, model, horizon=24, refit=False, eval_train=False):
    """
    Predicts the target variable for the testing data using an autoregressive approach,
    retraining the model after each window using the true values from the previous window.

    Args:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.
    - X_test (pd.DataFrame): Testing features.
    - y_test (pd.Series): Testing target.
    - model: Model to be trained and tested.
    - horizon (int): Number of hours to predict ahead.
    - refit (bool): Whether to refit the model after each window.
    - eval_train (bool): Whether to evaluate the model on the training set.

    Returns:
    - pd.Series: Contains the predictions from the model for the test set.
    - np.array | None: Coefficients for the linear regression model.
    """
    # Initialize prediction vector that will hold all predictions
    predictions = np.empty(len(X_test))

    # Prepare extended training set
    X_train_extended = X_train.copy()
    y_train_extended = y_train.copy()

    # Initial training of the model
    model.fit(X_train_extended, y_train_extended)

    # check if coefficients are present in case of linear regression model
    coeff = None
    if isinstance(model, LinearRegression):
        coeff = model.coef_
    print("Initial training completed with training size:", len(X_train_extended), "Indexes:", X_train_extended.index[0], "-", X_train_extended.index[-1])

    # Evaluate the model on the training set if eval_train is True
    if eval_train:
        return pd.Series(model.predict(X_train), index=X_train.index), coeff

    # Process each window
    num_windows = (len(X_test) + horizon - 1) // horizon

    for i in range(num_windows):
        start = i * horizon  
        end = min(start + horizon, len(X_test)) # Ensure that the last window is not larger than the test set
        
        # Print the range of indexes for current testing window
        print(f"Testing window {i}: Indexes {X_test.index[start]} - {X_test.index[end-1]}")

        # Prepare features for the current window
        X_test_current = X_test.iloc[start:end].copy()

        # Predict the current window
        current_predictions = model.predict(X_test_current)
        predictions[start:end] = current_predictions

        # Retrain the model with extended training set if refit is True
        if refit:
            # Append true outcomes to the training set (no need to copy data until re-training is needed)
            X_train_extended = pd.concat([X_train_extended, X_test_current])
            y_train_extended = pd.concat([y_train_extended, y_test.iloc[start:end]])
            model.fit(X_train_extended, y_train_extended)
            print(f"Retrained model with updated training size: {len(X_train_extended)}", "Indexes:", X_train_extended.index[0], "-", X_train_extended.index[-1])
            

    return pd.Series(predictions, index=X_test.index), coeff


def train_and_test_model(X_train, X_test, y_train, y_test, model, horizon=24, refit=False, eval_train=False):
    """
    Trains a model on the training data and evaluates it on the testing data.

    Args:
    - X_train (pd.DataFrame): Training features.
    - X_test (pd.DataFrame): Testing features.
    - y_train (pd.Series): Training target.
    - y_test (pd.Series): Testing target.
    - model: Model to be trained and tested.
    - horizon (int): Number of hours to predict ahead.
    - refit (bool): Whether to refit the model after each window.
    - eval_train (bool): Whether to evaluate the model on the training set.

    Returns:
    - dict: Contains the model, predictions, and evaluation metrics.
    """
  
    # Make predictions, maintaining index so that we can plot it correctly
    y_pred, coeff = ts_prediction(X_train, y_train, X_test, y_test, model, horizon, refit, eval_train)

    # Calculate evaluation metrics
    if eval_train:
        metrics = calculate_metrics(y_train, y_pred)
    else:
        metrics = calculate_metrics(y_test, y_pred)
    
    return {
        'model': model,
        'predictions': y_pred,
        'metrics': metrics,
        'coeff': coeff
    }   

def year_on_year_training(df, model):
    """
    Train the model year on year and return predictions and metrics.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    model: The model to be trained and tested.

    Returns:
    tuple: A tuple containing the predictions DataFrame and the metrics.
    """
    predictions = pd.Series()
    coeffs = {}

    # extract year from index
    years = df.index.year.unique()

    # fit year on year, train for one year predict for next etc for years 2016-2023
    for year in years[:-1]:

        print(f'Training model for year {year}')
        # Define the training data for the current year
        X_train = df.loc[df.index.year == year].drop(columns='y')
        y_train = df.loc[df.index.year == year]['y']
        X_test = df.loc[df.index.year == year + 1].drop(columns='y')
        y_test = df.loc[df.index.year == year + 1]['y']

        model.fit(X_train, y_train)
        y_hat_train = model.predict(X_train)
        y_hat_test = model.predict(X_test)

        predictions = pd.concat([predictions, y_hat_test])

        coeffs[year] = pd.Series(model.coeffs, index=X_train.columns)


    # Calculate the metrics for the Window Average model
    metrics = calculate_metrics(df['y'].loc[predictions.index], predictions)

    return predictions, metrics, coeffs