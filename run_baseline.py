from src.models.baseline_model import NaiveForecastModel
from src.data import UnivariatePreprocessor
from src.config import *
from src.utils import evaluate_model, execute_hyperparameter_tuning, setup_mlflow_experiment
import pandas as pd


if __name__=="__main__":
    setup_mlflow_experiment(MLFLOW_EXPERIMENT_NAME, tracking_uri=MLFLOW_URI)
    model = NaiveForecastModel()
    univariate_prep = UnivariatePreprocessor(
        timelag=1,
        test_size=0.2,
        scaler_type="standard"
    )
    sample_data = pd.read_csv('data/clean_BBCA_historical_data.csv')
    X_train, X_test, y_train, y_test = univariate_prep.fit_transform(sample_data)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    execute_hyperparameter_tuning(
        model_name="Baseline_Models",
        hyperparams_list=[{}],
        metrics_list=[metrics],
        test_identifier="BBCA",
        models_list=[model]
    )