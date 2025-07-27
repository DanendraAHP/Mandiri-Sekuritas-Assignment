from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd


class StatsModel:
    def __init__(self, data, exog_data, model_type):
        self.data = data
        self.exog_data = exog_data
        self.model = None
        self.model_type = model_type
        self.fitted_model = None
        self.order = None
        self.seasonal_order = None

    def set_parameters(self, order, seasonal_order=None):
        """Set ARIMA/SARIMAX parameters."""
        self.order = order
        if seasonal_order:
            self.seasonal_order = seasonal_order
    
    def build_model(self):
        """Build the ARIMA/SARIMAX model with set parameters."""
        if self.model_type=='ARIMA':
            if not self.order:
                raise ValueError("Parameters not set. Use set_parameters() first.")
            self.model = ARIMA(self.data, order=self.order, exog=self.exog_data)
        if self.model_type=='SARIMAX':
            if not self.order and not self.seasonal_order:
                raise ValueError("Parameters not set. Use set_parameters() first.")
            self.model = SARIMAX(self.data, order=self.order, seasonal_order=self.seasonal_order, exog=self.exog_data)

    def fit(self, **kwargs):
        """Fit the model."""
        if self.model is None:
            self.build_model()
        
        self.fitted_model = self.model.fit(**kwargs)
    
    def predict(self, steps: int, exog_data=None):
        """Make forecast."""
        if not self.fitted_model:
            raise ValueError("Model not fitted. Use fit() first.")
        forecast = self.fitted_model.get_forecast(steps=steps, exog=exog_data)
        forecast = pd.Series(forecast.predicted_mean).values
        return forecast