import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from forecaster.models import utils


class LinearModel:

    def __init__(self, 
                 daily_price_lags: list, 
                 time_features: bool=False,
                 external_features: dict={},
                 daily_external_lags: list=[],
                 fit_coeffs: bool=True,
                 ):
        """
        Initializes the ForecastingModel with the specified lags for daily prices and external features.
        """
        self.daily_price_lags = daily_price_lags
        self.time_features = time_features
        self.external_features = external_features
        self.daily_external_lags = daily_external_lags
        self.nFeatures = len(daily_price_lags) + time_features*48 + len(external_features) * len(daily_external_lags)
        self.coeffs = np.full(self.nFeatures, np.nan)
        self.fit_coeffs = fit_coeffs
        self.model = LinearRegression(fit_intercept=False)

    def preprocess_data(self, df):
        
        df_with_features = df.copy()

        if self.external_features:
            df_with_features = utils.add_external_features(df_with_features, self.external_features)

        if self.daily_price_lags:
            df_with_features = utils.create_daily_lag_features(
                df_with_features, 
                'y', 
                self.daily_price_lags, 
                average=True
                )

        if self.daily_external_lags:
            for feature_key in self.external_features.keys():
                feature_name = self.external_features[feature_key]
                df_with_features = utils.create_daily_lag_features(
                    df_with_features, 
                    feature_name, 
                    self.daily_external_lags, 
                    average=True
                    )

        if self.time_features:
            df_with_features = utils.extract_time_features(df_with_features)

        # remove rows with NaN values that might have been created by lagging
        df_with_features.dropna(how="any", inplace=True)

        return df_with_features

    def fit(self, X, y):
        # Fit coefficients useing least squares
        if self.fit_coeffs:
            self.model.fit(X, y)
            self.coeffs = self.model.coef_
        # Else just coeffs that sum to unity, only makes sense for models that only used lagged price feautures
        else:   
            self.coeffs = np.ones(self.nFeatures) / self.nFeatures

    def predict(self, X):
        y_hat = np.dot(X, self.coeffs)
        return pd.Series(y_hat, index=X.index, name="y_hat")