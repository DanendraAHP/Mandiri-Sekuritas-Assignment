import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional, List, Union
import warnings

def split_data_stats_model(data, train_percentage, used_columns):
    n_data = len(data)
    n_train = int(n_data*train_percentage)
    # Univariate
    if len(used_columns)==1:
        train = data.iloc[:n_train][used_columns[0]]
        test = data.iloc[n_train:][used_columns[0]]
        return train, test
    # Multivariate
    else:
        target_series = data[['Close']]
        exog_data = data[used_columns].drop(columns = ['Close'])
        train_target = target_series[:n_train]
        test_target = target_series[n_train:]

        train_exog = exog_data[:n_train]
        test_exog = exog_data[n_train:]
        return train_target, train_exog, test_target, test_exog

class BasePreprocessor:
    """Base class for stock data preprocessing."""
    
    def __init__(self, 
                 timelag: int = 5,
                 test_size: float = 0.2,
                 scaler_type: str = 'standard',
                 target_column: str = 'Close'):
        """
        Initialize base preprocessor.
        
        Args:
            timelag (int): Number of previous days to use for prediction
            test_size (float): Proportion of data to use for testing (0.0 to 1.0)
            scaler_type (str): Type of scaler ('standard', 'minmax', or 'none')
            target_column (str): Name of the target column to predict
        """
        self.timelag = timelag
        self.test_size = test_size
        self.scaler_type = scaler_type
        self.target_column = target_column
        self.scaler = None
        self.target_scaler = None
        self.feature_columns = None
        
        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
        elif scaler_type == 'none':
            self.scaler = None
            self.target_scaler = None
        else:
            raise ValueError("scaler_type must be 'standard', 'minmax', or 'none'")
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean input data."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        # Check for missing values
        if data.isnull().sum().sum() > 0:
            warnings.warn("Missing values detected. Filling with forward fill method.")
            data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure data is sorted by date/index
        if not data.index.is_monotonic_increasing:
            warnings.warn("Data is not sorted by index. Sorting now.")
            data = data.sort_index()
        
        return data
    
    def _create_sequences(self, 
                         features: np.ndarray, 
                         target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            features (np.ndarray): Feature data
            target (np.ndarray): Target data
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X sequences and y targets
        """
        X, y = [], []
        
        for i in range(self.timelag, len(features)):
            # Get the last 'timelag' days as features
            X.append(features[i-self.timelag:i])
            # Get the next day target
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def _split_data(self, 
                   X: np.ndarray, 
                   y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets maintaining temporal order.
        
        Args:
            X (np.ndarray): Feature sequences
            y (np.ndarray): Target values
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        # Calculate split point
        split_idx = int(len(X) * (1 - self.test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled target values back to original scale.
        
        Args:
            y_scaled (np.ndarray): Scaled target values
            
        Returns:
            np.ndarray: Original scale target values
        """
        if self.target_scaler is not None:
            if y_scaled.ndim == 1:
                y_scaled = y_scaled.reshape(-1, 1)
            return self.target_scaler.inverse_transform(y_scaled).flatten()
        return y_scaled


class UnivariatePreprocessor(BasePreprocessor):
    """
    Preprocessor for univariate time series forecasting.
    Uses only the target variable (e.g., closing price).
    """
    
    def fit_transform(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit preprocessor and transform data for univariate forecasting.
        
        Args:
            data (pd.DataFrame): Input stock data with target column
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        # Validate data
        data = self._validate_data(data)
        
        # Extract target variable
        target_data = data[self.target_column].values.reshape(-1, 1)
        
        # Fit and transform target data
        if self.target_scaler is not None:
            target_scaled = self.target_scaler.fit_transform(target_data).flatten()
        else:
            target_scaled = target_data.flatten()
        
        # Create sequences (for univariate, features and target are the same)
        X, y = self._create_sequences(target_scaled.reshape(-1, 1), target_scaled)
        
        # Flatten X for univariate case (timelag, 1) -> (timelag,)
        X = X.squeeze(axis=-1)
        
        # Split data
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        
        return X_train, X_test, y_train, y_test
    
    def transform(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            data (pd.DataFrame): New data to transform
            
        Returns:
            Tuple: X, y sequences
        """
        if self.target_scaler is None and self.scaler_type != 'none':
            raise ValueError("Preprocessor must be fitted before transform")
        
        data = self._validate_data(data)
        target_data = data[self.target_column].values.reshape(-1, 1)
        
        if self.target_scaler is not None:
            target_scaled = self.target_scaler.transform(target_data).flatten()
        else:
            target_scaled = target_data.flatten()
        
        X, y = self._create_sequences(target_scaled.reshape(-1, 1), target_scaled)
        X = X.squeeze(axis=-1)
        
        return X, y


class MultivariatePreprocessor(BasePreprocessor):
    """
    Preprocessor for multivariate time series forecasting.
    Uses multiple features (open, high, low, close, volume, etc.).
    """
    
    def __init__(self, 
                 timelag: int = 5,
                 test_size: float = 0.2,
                 scaler_type: str = 'standard',
                 target_column: str = 'Close',
                 feature_columns: Optional[List[str]] = None):
        """
        Initialize multivariate preprocessor.
        
        Args:
            timelag (int): Number of previous days to use for prediction
            test_size (float): Proportion of data to use for testing
            scaler_type (str): Type of scaler ('standard', 'minmax', or 'none')
            target_column (str): Name of the target column to predict
            feature_columns (List[str], optional): List of feature columns to use.
                                                 If None, uses all numeric columns except target
        """
        super().__init__(timelag, test_size, scaler_type, target_column)
        self.feature_columns = feature_columns
    
    def _select_features(self, data: pd.DataFrame) -> List[str]:
        """Select feature columns for multivariate modeling."""
        if self.feature_columns is not None:
            # Use specified feature columns
            missing_cols = [col for col in self.feature_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Feature columns not found in data: {missing_cols}")
            return self.feature_columns
        else:
            # Use all numeric columns except target
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_columns if col != self.target_column]
            
            if len(feature_cols) == 0:
                raise ValueError("No feature columns found. Please specify feature_columns.")
            
            return feature_cols
    
    def fit_transform(self, 
                     data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit preprocessor and transform data for multivariate forecasting.
        
        Args:
            data (pd.DataFrame): Input stock data
            add_indicators (bool): Whether to add technical indicators
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        # Validate data
        data = self._validate_data(data)
        
        # Select feature columns
        self.feature_columns = self._select_features(data)
        
        # Extract features and target
        features = data[self.feature_columns].values
        target = data[self.target_column].values
        
        # Fit and transform features
        if self.scaler is not None:
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = features
        
        # Fit and transform target
        if self.target_scaler is not None:
            target_scaled = self.target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
        else:
            target_scaled = target
        
        # Create sequences
        X, y = self._create_sequences(features_scaled, target_scaled)
        
        # Split data
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        
        return X_train, X_test, y_train, y_test
    
    def transform(self, data: pd.DataFrame, add_indicators: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            data (pd.DataFrame): New data to transform
            add_indicators (bool): Whether to add technical indicators
            
        Returns:
            Tuple: X, y sequences
        """
        if self.scaler is None and self.scaler_type != 'none':
            raise ValueError("Preprocessor must be fitted before transform")
        
        if self.feature_columns is None:
            raise ValueError("Preprocessor must be fitted before transform")
        
        data = self._validate_data(data)
        
        if add_indicators:
            data = self.add_technical_indicators(data)
        
        features = data[self.feature_columns].values
        target = data[self.target_column].values
        
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
        
        if self.target_scaler is not None:
            target_scaled = self.target_scaler.transform(target.reshape(-1, 1)).flatten()
        else:
            target_scaled = target
        
        X, y = self._create_sequences(features_scaled, target_scaled)
        
        return X, y

