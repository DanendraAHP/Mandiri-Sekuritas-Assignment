import mlflow
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any
import random
from functools import partial
from itertools import starmap
from more_itertools import consume


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance using multiple regression metrics.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics
            - mae: Mean Absolute Error
            - mse: Mean Squared Error  
            - rmse: Root Mean Squared Error
            - r2: R-squared (coefficient of determination)
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Calculate metrics using sklearn
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Return as dictionary
    return {
        'mae': float(mae),
        'mse': float(mse), 
        'rmse': float(rmse),
        'r2': float(r2)
    }


def log_run(run_name, model_type, hyperparams, metrics, tag_ident, model_object=None, input_example=None):
    """
    Log individual hyperparameter run as child run WITH MODEL SAVING.
    """
    with mlflow.start_run(run_name=run_name, nested=True):
        # Log hyperparameters
        mlflow.log_params(hyperparams)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Add tags
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("test_identifier", tag_ident)
        
        # 🔥 FIXED: Save model if provided
        if model_object is not None:
            try:
                if model_type in ['XGBoost', 'LightGBM']:
                    # Fix both warnings
                    mlflow.sklearn.log_model(
                        sk_model=model_object, 
                        artifact_path=f"{model_type}_model",  # Keep this for compatibility
                        input_example=input_example[:5] if input_example is not None else None  # Fix signature warning
                    )
                else:  # ARIMA, SARIMAX
                    import pickle
                    import tempfile
                    import os
                    
                    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
                        pickle.dump(model_object, tmp_file)
                        tmp_file.flush()
                        mlflow.log_artifact(tmp_file.name, f"{model_type}_model")
                    
                    os.unlink(tmp_file.name)
                
                mlflow.set_tag("model_saved", "True")
                print(f"    ✅ Model saved for {run_name}")
                
            except Exception as e:
                mlflow.set_tag("model_saved", "False")
                mlflow.set_tag("save_error", str(e))
                print(f"    ❌ Model save failed for {run_name}: {e}")
        else:
            mlflow.set_tag("model_saved", "False")


def execute_hyperparameter_tuning(
    model_name,
    hyperparams_list,
    metrics_list,
    test_identifier="",
    models_list=None
):
    """
    Execute hyperparameter tuning and log experiments to MLflow.
    
    Args:
        model_name (str): Name for the MLflow experiment
        hyperparams_list (list): List of hyperparameter dictionaries
        metrics_list (list): List of metrics dictionaries from evaluate_model()
        test_identifier (str, optional): Additional identifier for run names
        models_list (list, optional): List of trained model objects to log
        
    Example:
        >>> hyperparams = [{"window": 5}, {"window": 10}]
        >>> metrics = [{"rmse": 200.5, "mae": 150.3, "r2": 0.8, "mape": 3.2}]
        >>> execute_hyperparameter_tuning("Baseline_Models", hyperparams, metrics)
    """
    ident = "default" if not test_identifier else test_identifier
    
    with mlflow.start_run(run_name=f"parent_{model_name}_{ident}"):
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("test_identifier", ident)
        mlflow.set_tag("run_type", "parent")
        
        best_rmse = float('inf')
        best_run_name = None
        models_saved_count = 0
        
        # 🔥 NEW: Generate run names with column info
        for i, (hyperparams, metrics) in enumerate(zip(hyperparams_list, metrics_list)):
            
            # Create descriptive run name with columns
            if 'used_columns' in hyperparams:
                columns = hyperparams['used_columns']
                col_str = "_".join(columns)
            else:
                col_str = f"col{i}"
            
            # Create base run name (your existing logic)
            if model_name in ['ARIMA', 'SARIMAX']:
                if 'order' in hyperparams:
                    order_str = "_".join(map(str, hyperparams['order']))
                    if 'seasonal_order' in hyperparams and hyperparams['seasonal_order'] != (0,0,0,0):
                        seasonal_str = "s" + "_".join(map(str, hyperparams['seasonal_order']))
                        param_str = f"{order_str}_{seasonal_str}"
                    else:
                        param_str = order_str
                else:
                    param_str = f"run_{i}"
            else:  # XGBoost, LightGBM
                key_params = []
                if 'timelag' in hyperparams:
                    key_params.append(f"t{hyperparams['timelag']}")
                if 'n_estimators' in hyperparams:
                    key_params.append(f"n{hyperparams['n_estimators']}")
                if 'max_depth' in hyperparams:
                    key_params.append(f"d{hyperparams['max_depth']}")
                param_str = "_".join(key_params) if key_params else f"run_{i}"
            
            # 🔥 COMBINE: model + params + columns
            run_name = f"{model_name}_{param_str}_{col_str}"
            print("RUN NAME CREATED")
            print(run_name)
            # Get model object if available
            model_object = None
            if models_list and i < len(models_list):
                model_object = models_list[i]
            
            # Log the run (same as your existing log_run function)
            log_run(run_name, model_name, hyperparams, metrics, ident, model_object)
            
            # Track best
            if metrics.get('rmse', float('inf')) < best_rmse:
                best_rmse = metrics['rmse']
                best_run_name = run_name
            
            if model_object is not None:
                models_saved_count += 1
        
        # Log summary
        mlflow.log_metric("best_rmse", best_rmse)
        mlflow.log_metric("total_runs", len(hyperparams_list))
        mlflow.log_metric("models_saved", models_saved_count)
        mlflow.log_param("best_run", best_run_name)
        
        print(f"📊 {model_name} Summary: Best RMSE: {best_rmse:.4f}, Best Run: {best_run_name}")


def generate_run_names(model_name, hyperparams_list):
    """
    Generate run names based on hyperparameters.
    
    Args:
        model_name (str): Name of the model
        hyperparams_list (list): List of hyperparameter dictionaries
        
    Returns:
        generator: Generator of run names
    """
    for i, params in enumerate(hyperparams_list):
        # Create descriptive run name based on parameters
        if model_name in ['ARIMA', 'SARIMAX']:
            if 'order' in params:
                order_str = f"{'_'.join(map(str, params['order']))}"
                if 'seasonal_order' in params and params['seasonal_order'] != (0,0,0,0):
                    seasonal_str = f"s{'_'.join(map(str, params['seasonal_order']))}"
                    param_str = f"{order_str}_{seasonal_str}"
                else:
                    param_str = order_str
            else:
                param_str = f"run_{i}"
        else:  # XGBoost, LightGBM
            # Use key parameters for name
            key_params = []
            if 'timelag' in params:
                key_params.append(f"t{params['timelag']}")
            if 'n_estimators' in params:
                key_params.append(f"n{params['n_estimators']}")
            if 'max_depth' in params:
                key_params.append(f"d{params['max_depth']}")
            
            param_str = "_".join(key_params) if key_params else f"run_{i}"
        
        yield f"{model_name}_{param_str}"


def setup_mlflow_experiment(experiment_name="stock_forecasting", tracking_uri=None):
    """
    Setup MLflow experiment and tracking.
    
    Args:
        experiment_name (str): Name of the experiment
        tracking_uri (str): MLflow tracking URI (optional)
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.set_experiment(experiment_name)