from src.data import UnivariatePreprocessor, MultivariatePreprocessor, split_data_stats_model
from src.models import StatsModel, MODEL_CONFIG
import xgboost as xgb
import lightgbm as lgb
from src.utils import *
from src.config import *

import pandas as pd
import warnings
warnings.filterwarnings('ignore')  # Suppress ARIMA convergence warnings

def train_eval_model():
    setup_mlflow_experiment(MLFLOW_EXPERIMENT_NAME, tracking_uri=MLFLOW_URI)
    data = pd.read_csv(DATA_PATH)
    col_combination = create_incremental_combinations(USED_COLUMNS)
    
    for model_name in MODEL_CONFIG:  # model_name is a string like 'ARIMA'
        print(f"\n{'='*50}")
        print(f"TRAINING {model_name}")
        print(f"{'='*50}")
        
        try:
            # Collect ALL results for this model first
            all_hyperparams = []
            all_metrics = []
            all_models = []

            for params in MODEL_CONFIG[model_name]:
                print(f"\nTesting parameters: {params}")
                for col in col_combination:
                    try:
                        print(f'  Columns: {col}')
                        
                        if model_name in ['ARIMA', 'SARIMAX']:
                            # ARIMA/SARIMAX handling
                            if len(col) == 1:
                                train, y_test = split_data_stats_model(data, TRAIN_PERCENTAGE, col)
                                model = StatsModel(train, None, model_name)
                            else:
                                train_target, train_exog, y_test, test_exog = split_data_stats_model(data, TRAIN_PERCENTAGE, col)
                                model = StatsModel(train_target, train_exog, model_name)
                            
                            if model_name == 'ARIMA':
                                model.set_parameters(order=params['order'])
                            elif model_name == 'SARIMAX':
                                model.set_parameters(order=params['order'], seasonal_order=params['seasonal_order'])
                            
                            model.fit()
                            
                            if len(col) == 1:
                                y_pred = model.predict(steps=len(y_test))
                            else:
                                y_pred = model.predict(steps=len(y_test), exog_data=test_exog)
                        
                        else:
                            # XGBoost/LightGBM handling
                            if len(col) == 1:
                                # Univariate
                                univariate_prep = UnivariatePreprocessor(
                                    timelag=params['timelag'],
                                    test_size=1-TRAIN_PERCENTAGE,
                                    scaler_type=params['scaler_type']
                                )
                                X_train, X_test, y_train, y_test = univariate_prep.fit_transform(data)
                            else:
                                # Multivariate
                                multivariate_prep = MultivariatePreprocessor(
                                    timelag=params['timelag'],
                                    test_size=1-TRAIN_PERCENTAGE,
                                    scaler_type=params['scaler_type'],
                                    feature_columns=col,
                                )
                                X_train, X_test, y_train, y_test = multivariate_prep.fit_transform(data)
                                # Remove these lines:
                                X_train = X_train.reshape(X_train.shape[0], -1)
                                X_test = X_test.reshape(X_test.shape[0], -1)
                            
                            # Remove timelag and scaler_type from model parameters
                            model_params = {k: v for k, v in params.items() if k not in ['timelag', 'scaler_type']}
                            
                            if model_name == 'XGBoost':
                                model = xgb.XGBRegressor(random_state=RANDOM_STATE, **model_params)
                            elif model_name == 'LightGBM':
                                model = lgb.LGBMRegressor(random_state=RANDOM_STATE, **model_params)
                            
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                        
                        # Evaluate
                        model_eval = evaluate_model(y_test, y_pred)
                        # param_results.append(model_eval)
                        print(f'    RMSE: {model_eval["rmse"]:.4f}')

                        # Log the model
                        parsed_params = {**params}
                        parsed_params['used_columns'] = col
                        print(f'  Columns: {col} params appended')
                        print(parsed_params)
                        all_hyperparams.append(parsed_params)
                        all_metrics.append(model_eval)
                        all_models.append(model)

                        
                    except Exception as e:
                        print(f'    Error: {str(e)}')
             
        except Exception as e:
            print(f'Model {model_name} failed: {str(e)}')
            import traceback
            traceback.print_exc()
        print(all_hyperparams)
        execute_hyperparameter_tuning(
            model_name=model_name,
            hyperparams_list=all_hyperparams,
            metrics_list=all_metrics,
            test_identifier="BBCA",
            models_list=all_models  # ← Just add this line!
        )

if __name__ == '__main__':
    train_eval_model()