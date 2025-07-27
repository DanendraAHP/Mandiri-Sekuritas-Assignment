import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.base import BaseEstimator, RegressorMixin


class NaiveForecastModel(BaseEstimator, RegressorMixin):
    """
    Naive forecast baseline model where the next day's price equals the current day's price.
    """
    
    def __init__(self):
        self.model_name = "Naive_Forecast"
        self.last_value_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaiveForecastModel':
        """
        Fit the naive model by storing the last known value.
        
        Args:
            X: Feature matrix (not used in naive forecast)
            y: Target values
            
        Returns:
            self: Fitted model
        """
        self.last_value_ = y[-1]  # Store the last value for prediction
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using naive forecast (last known value).
        
        Args:
            X: Feature matrix
            
        Returns:
            np.ndarray: Predictions (all equal to last known value)
        """
        if self.last_value_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return np.full(X.shape[0], self.last_value_)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {"model_name": self.model_name}
    
    def set_params(self, **params) -> 'NaiveForecastModel':
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self