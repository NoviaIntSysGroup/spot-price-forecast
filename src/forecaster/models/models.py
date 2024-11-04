import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

    
class WindowAverageModel:
    """
    A model that predicts the average of the last 'window_size' observations for all future values.
    """
    def __init__(self, window_size=24):
        self.window_size = window_size
        self.window_average = None
    
    def fit(self, X, y):
        if hasattr(y, 'iloc'):  # Pandas series handling
            self.window_average = y.iloc[-self.window_size:].mean()
        else:  # Handling numpy arrays or Python lists
            self.window_average = np.mean(y[-self.window_size:])
    
    def predict(self, X):
        return np.full(len(X), self.window_average)
    
class ExponentialAverageModel:
    """
    A model that predicts the exponential average for all future values.
    """
    def __init__(self, alpha=0.1):
        """
        Initializes the ExponentialAverage with a specific alpha value.
        
        Args:
            alpha (float): The smoothing factor for the exponential average.
        """
        self.alpha = alpha
        self.exponential_average = None
    
    def fit(self, X, y):
        """
        Fits the model using the training data to compute the exponential average.
        
        Args:
            X: Features data (not used in the computation as the model is based only on y)
            y: Target data from which the exponential average is computed.
        """
        self.exponential_average = y.ewm(alpha=self.alpha, adjust=False).mean().iloc[-1]
    
    def predict(self, X):
        """
        Predicts using the computed exponential average for each entry in X.
        
        Args:
            X: Data for which predictions are to be made (not used in predictions as the output is always the average).
        
        Returns:
            numpy.ndarray: An array filled with the computed average, with the same length as X.
        """
        return np.full(len(X), self.exponential_average)