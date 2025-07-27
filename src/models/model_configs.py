# Model configurations for stock price forecasting
MODEL_CONFIG = {
    'ARIMA': [
        {'order': (1, 1, 1)},  # Simple and robust
        {'order': (2, 1, 1)},  # Standard choice
        {'order': (1, 1, 2)},  # Good alternative
        {'order': (2, 1, 2)},  # Balanced complexity
        {'order': (3, 1, 2)},  # Higher complexity
    ],
    
    'SARIMAX': [
        {'order': (2, 1, 1), 'seasonal_order': (0, 0, 0, 0)},   # Non-seasonal baseline
        {'order': (1, 1, 1), 'seasonal_order': (1, 0, 1, 5)},   # Weekly seasonality
        {'order': (2, 1, 2), 'seasonal_order': (1, 0, 1, 5)},   # Complex weekly
        {'order': (1, 1, 1), 'seasonal_order': (1, 0, 1, 22)},  # Monthly seasonality
        {'order': (2, 1, 1), 'seasonal_order': (1, 0, 1, 22)},  # Complex monthly
    ],
    
    'XGBoost': [
        # timelag: 1, 5, 10, 15, 22 days
        # scaler_type: 'standard' or 'minmax'
        # NO random_state parameter
        {
            'timelag': 1,
            'scaler_type': 'standard',
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        {
            'timelag': 5,
            'scaler_type': 'minmax',
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.08,
            'subsample': 0.85,
            'colsample_bytree': 0.85
        },
        {
            'timelag': 10,
            'scaler_type': 'standard',
            'n_estimators': 150,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.05,
            'reg_lambda': 0.05
        },
        {
            'timelag': 15,
            'scaler_type': 'minmax',
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.9
        },
        {
            'timelag': 22,
            'scaler_type': 'standard',
            'n_estimators': 250,
            'max_depth': 10,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.01,
            'reg_lambda': 0.01
        }
    ],
    
    'LightGBM': [
        # timelag: 1, 5, 10, 15, 22 days
        # scaler_type: 'standard' or 'minmax'
        # NO random_state parameter
        {
            'timelag': 1,
            'scaler_type': 'standard',
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbose': -1
        },
        {
            'timelag': 5,
            'scaler_type': 'minmax',
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.08,
            'num_leaves': 100,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'verbose': -1
        },
        {
            'timelag': 10,
            'scaler_type': 'standard',
            'n_estimators': 150,
            'max_depth': 8,
            'learning_rate': 0.1,
            'num_leaves': 150,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.05,
            'reg_lambda': 0.05,
            'verbose': -1
        },
        {
            'timelag': 15,
            'scaler_type': 'minmax',
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'num_leaves': 120,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'verbose': -1
        },
        {
            'timelag': 22,
            'scaler_type': 'standard',
            'n_estimators': 250,
            'max_depth': 10,
            'learning_rate': 0.03,
            'num_leaves': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.01,
            'reg_lambda': 0.01,
            'verbose': -1
        }
    ]
}

def get_model_configs():
    """Return the model configurations."""
    return MODEL_CONFIG