# Stock Price Forecasting Technical Test

## Overview

This project implements a machine learning model to forecast the closing price of BBCA (Bank Central Asia) stock for the next trading day. The solution demonstrates MLOps principles through integration with MLflow for experiment tracking and model management.

## Objective

Develop a regression-based machine learning model that predicts the actual closing price of BBCA stock for the next trading day, with comprehensive MLOps integration for experiment tracking, model logging, and reproducibility.

## Project Structure

```
MANDIRI_TEST/
├── data/                           # Data storage
│   ├── BBCA_historical_data.csv    # Raw historical stock data
│   └── clean_BBCA_historical_data.csv  # Preprocessed data
├── notebooks/                      # Jupyter notebooks
│   └── eda.ipynb                   # Exploratory Data Analysis
├── src/                           # Source code modules
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py         # Data loading utilities
│   │   └── preprocessor.py        # Data preprocessing pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_model.py      # Baseline model implementation
│   │   ├── model_configs.py       # Model configuration settings
│   │   └── regression_model.py    # Main regression model
│   └── utils/
│       ├── __init__.py
│       ├── column_combinations.py # Feature engineering utilities
│       └── send_metric_mlflow.py  # MLflow integration functions
├── main.py                        # Main execution script
├── config.py                      # Configuration settings
├── requirements.txt               # Project dependencies
├── test.ipynb                     # Testing notebook
└── README.md                      # This file
```

## Setup Instructions

### Prerequisites
- Python 3.11
- Conda package manager

### Environment Setup

1. **Clone/Extract the project:**
   ```bash
   # Extract the project folder
   cd MANDIRI_TEST
   ```

2. **Create Conda Environment:**
   ```bash
   conda create -n stock_forecast python=3.11
   conda activate stock_forecast
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Required Dependencies
- pandas
- numpy
- scikit-learn
- yfinance
- mlflow
- matplotlib
- seaborn
- jupyter

## How to Run

### Quick Start
```bash
# Activate environment
conda activate stock_forecast

# Run the complete pipeline
python main.py
```

This will:
1. Load and preprocess the BBCA historical stock data
2. Train both baseline and regression models
3. Evaluate model performance
4. Log experiments to MLflow
5. Generate predictions for the next trading day

### Exploratory Data Analysis
To understand the data preprocessing steps:
```bash
jupyter notebook notebooks/eda.ipynb
```

The EDA notebook contains detailed analysis of:
- Data quality assessment
- Missing value handling
- Feature engineering process
- Technical indicator creation
- Data visualization

## Model Approach

### Regression-Based Forecasting
- **Objective:** Predict the actual closing price for the next trading day
- **Algorithm:** Advanced regression model (implementation in `src/models/regression_model.py`)
- **Features:** 
  - Lagged price variables
  - Rolling averages (5, 10, 20 days)
  - Technical indicators (RSI, MACD, Bollinger Bands)
  - Volume-based features
  - Volatility measures

### Baseline Model
- **Type:** Naive forecast (next day price = current day price)
- **Purpose:** Performance comparison benchmark
- **Implementation:** `src/models/baseline_model.py`

## MLOps Integration (MLflow)

### Experiment Tracking
- **Metrics Logged:** RMSE, MAE, R-squared, MAPE
- **Parameters Logged:** Model hyperparameters, feature sets, preprocessing steps
- **Artifacts Logged:** Trained models, preprocessing pipelines, performance plots

### Model Management
- Automatic model registration based on performance metrics
- Model versioning and comparison
- Reproducible experiment runs

### Accessing MLflow UI
```bash
mlflow ui
```
Visit `http://localhost:5000` to view:
- Experiment runs and comparisons
- Model performance metrics
- Logged artifacts and parameters
- Model registry and versions

## Key Features

### Data Processing
- **Data Source:** BBCA historical stock data via yfinance
- **Preprocessing:** Automated cleaning pipeline (see `notebooks/eda.ipynb` for details)
- **Feature Engineering:** Technical indicators and lagged variables
- **Missing Value Handling:** Forward fill and interpolation methods

### Model Development
- **Training Period:** Historical data with train/validation split
- **Feature Selection:** Automated feature importance ranking
- **Hyperparameter Tuning:** Grid search with cross-validation
- **Evaluation:** Multiple regression metrics for comprehensive assessment

### Code Quality
- **Modular Design:** Separated concerns across logical modules
- **Error Handling:** Comprehensive exception handling
- **Documentation:** Well-commented code with docstrings
- **Reproducibility:** Fixed random seeds and logged configurations

## Results and Evaluation

### Performance Metrics
The model is evaluated using:
- **RMSE (Root Mean Square Error):** Primary metric for model selection
- **MAE (Mean Absolute Error):** Average prediction error
- **R-squared:** Explained variance
- **MAPE (Mean Absolute Percentage Error):** Relative error percentage

### Model Comparison
Performance comparison between:
1. Baseline model (naive forecast)
2. Regression model with engineered features
3. Model performance tracked in MLflow for easy comparison

## Reproduction Instructions

1. **Environment Setup:**
   ```bash
   conda create -n stock_forecast python=3.11
   conda activate stock_forecast
   pip install -r requirements.txt
   ```

2. **Run Complete Pipeline:**
   ```bash
   python main.py
   ```

3. **View Results:**
   ```bash
   mlflow ui  # Access MLflow dashboard
   ```

4. **Explore Analysis:**
   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```

## Data Source

- **Stock:** BBCA (Bank Central Asia) - Indonesian Banking Stock
- **Data Provider:** Yahoo Finance via `yfinance` library
- **Period:** Historical data with sufficient lookback for feature engineering
- **Features:** OHLCV (Open, High, Low, Close, Volume) data

## Technical Implementation

### Data Pipeline
1. **Loading:** `src/data/data_loader.py` - Handles data ingestion
2. **Preprocessing:** `src/data/preprocessor.py` - Cleans and engineers features
3. **Configuration:** `config.py` - Centralized settings management

### Model Pipeline
1. **Baseline:** `src/models/baseline_model.py` - Simple benchmark model
2. **Regression:** `src/models/regression_model.py` - Advanced ML model
3. **Configuration:** `src/models/model_configs.py` - Model hyperparameters

### MLOps Pipeline
1. **Tracking:** `src/utils/send_metric_mlflow.py` - MLflow integration
2. **Utilities:** `src/utils/column_combinations.py` - Feature engineering helpers

## Next Steps

- Model can be deployed for real-time predictions
- MLflow model registry enables easy model versioning
- Pipeline can be extended with additional technical indicators
- Automated retraining can be implemented with MLflow

## MLflow Screenshots

The following screenshots demonstrate the MLOps integration and experiment tracking capabilities:

### 1. Experiment Overview
![Parent Experiment Runs](mlflow_screenshots/Parent%20Experiment%20Runs.png)
*Complete view of all experiment runs showing both baseline and regression model experiments with timestamps and run IDs for full traceability.*

### 2. Model Performance Comparison
![Sorted Runs by RMSE](mlflow_screenshots/Sorted%20Run%20by%20RMSE.png)
*Experiments sorted by RMSE performance metric, clearly demonstrating the superior performance of the regression model compared to the baseline model.*

### 3. Best Model Artifacts
![Best Run Artifacts](mlflow_screenshots/Best%20Run%20Artifact.png)
*Detailed view of logged artifacts for the best performing model, including model files, preprocessing pipelines, performance visualizations, and feature importance plots.*

### 4. Model Registry
![Model Registry](mlflow_screenshots/Model%20Registry.png)
*Model management dashboard showing registered models with versions, stages, and metadata for production-ready model deployment.*

## Contact

For questions or issues, please contact the development team or refer to the technical documentation in the code comments.