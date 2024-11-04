from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error


def calculate_metrics(y_test, y_pred):
    """
    Calculates the mean squared error, mean absolute error, and root mean squared error.

    Args:
    - y_test (np.array): True target values.
    - y_pred (np.array): Predicted target values.

    Returns:
    - dict: Contains the mean squared error, mean absolute error, and root mean squared error.
    """
    
    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate Root Mean Squared Error
    rmse = root_mean_squared_error(y_test, y_pred)

    # Return a dictionary with all three metrics
    return {
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
        'root_mean_squared_error': rmse
    }