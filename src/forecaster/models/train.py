import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from forecaster.models.evaluate import calculate_metrics


def year_on_year_training(df, model):
    """
    Train the model year on year and return predictions and metrics.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    model: The model to be trained and tested.

    Returns:
    tuple: A tuple containing the predictions DataFrame and the metrics.
    """
    
    coeffs = {}
    predictions_test = []
    predictions_train = []

    # extract year from index
    years = df.index.year.unique()

    # fit year on year, train for one year predict for next etc for years 2016-2023
    for year in years[:-1]:

        print(f'Training model for year {year}')
        # Define the training data for the current year
        X_train = df.loc[df.index.year == year].drop(columns='y')
        y_train = df.loc[df.index.year == year]['y']
        X_test = df.loc[df.index.year == year + 1].drop(columns='y')

        model.fit(X_train, y_train)
        y_hat_train = model.predict(X_train)
        y_hat_test = model.predict(X_test)

        predictions_test.append(y_hat_test)
        predictions_train.append(y_hat_train)

        coeffs[year] = pd.Series(model.coeffs, index=X_train.columns)

    predictions = {
        'test': pd.concat(predictions_test),
        'train': pd.concat(predictions_train)
    }

    # Calculate the metrics for the Window Average model
    metrics = {
        'test': calculate_metrics(df['y'].loc[predictions['test'].index], predictions['test']),
        'train': calculate_metrics(df['y'].loc[predictions['train'].index], predictions['train'])
    }

    return predictions, metrics, coeffs