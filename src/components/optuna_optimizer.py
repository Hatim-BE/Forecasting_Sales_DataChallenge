import numpy as np
import optuna
import os
import sys
import pickle
optuna.logging.set_verbosity(optuna.logging.WARNING)
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
from src.components.data_ingestion import DataIngestion
from src.components.data_tranformation import DataTransformation
from src.utils import *
from src.exception import CustomException
from src.logger import logging

# Models and Parameters Dictionary
models_params = {
    # "Linear Regression": {
    #     "model": LinearRegression(),
    #     "params": {}
    # },

    # "CatBoost": {
    #     "model": None,  # Placeholder for dynamic instantiation
    #     "params": {
    #         "iterations": lambda trial: trial.suggest_int("iterations", 100, 1000, step=100),
    #         "learning_rate": lambda trial: trial.suggest_float("learning_rate", 0.01, 0.3),
    #         "depth": lambda trial: trial.suggest_int("depth", 4, 10),
    #         "l2_leaf_reg": lambda trial: trial.suggest_float("l2_leaf_reg", 1e-2, 10.0),
    #         "subsample": lambda trial: trial.suggest_float("subsample", 0.6, 1.0),
    #         "random_strength": lambda trial: trial.suggest_float("random_strength", 0.0, 10.0),
    #     }
    # },
    # "Ridge": {
    #     "model": Ridge(),
    #     "params":{
    #         "alpha": lambda trial: trial.suggest_float("alpha", 1e-3, 10.0, log=True)
    #     }
    # },

    # "Lasso": {
    #     "model": Lasso(),
    #     "params": {
    #         "alpha": lambda trial: trial.suggest_float("alpha", 0.01, 1.0),
    #         "max_iter": lambda trial: trial.suggest_int("max_iter", 500, 2000, step=500),
    #         "tol": lambda trial: trial.suggest_float("tol", 1e-4, 1e-2)
    #     }
    # },

    "XGBoost Regressor": {
        "model": XGBRegressor(),
        "params": {
            "n_estimators": lambda trial: trial.suggest_int('n_estimators', 100, 1000),
            "learning_rate": lambda trial: trial.suggest_float('learning_rate', 0.01, 0.3),
            "max_depth": lambda trial: trial.suggest_int('max_depth', 3, 10),
            "subsample": lambda trial: trial.suggest_float('subsample', 0.5, 1.0),
            "colsample_bytree": lambda trial: trial.suggest_float('colsample_bytree', 0.5, 1.0),
            "alpha": lambda trial: trial.suggest_float('reg_alpha', 1e-4, 10.0),
            "reg_lambda": lambda trial: trial.suggest_float('reg_lambda', 1e-4, 10.0)
        }
    },

    # "LightGBM Regressor": {
    #     "model": LGBMRegressor(verbose=-1),
    #     "params": {
    #         "n_estimators": lambda trial: trial.suggest_int("n_estimators", 50, 200, step=50),
    #         "learning_rate": lambda trial: trial.suggest_float("learning_rate", 0.01, 0.2),
    #         "max_depth": lambda trial: trial.suggest_int("max_depth", -1, 10),
    #         "num_leaves": lambda trial: trial.suggest_int("num_leaves", 31, 100),
    #         "subsample": lambda trial: trial.suggest_float("subsample", 0.8, 1.0),
    #         "colsample_bytree": lambda trial: trial.suggest_float("colsample_bytree", 0.8, 1.0)
    #     }
    # },

    # "Gradient Boosting": {
    #     "model": GradientBoostingRegressor(),
    #     "params": {
    #         "n_estimators": lambda trial: trial.suggest_int("n_estimators", 50, 200, step=50),
    #         "learning_rate": lambda trial: trial.suggest_float("learning_rate", 0.01, 0.2),
    #         "min_samples_split": lambda trial: trial.suggest_int("min_samples_split", 2, 10),
    #         "subsample": lambda trial: trial.suggest_float("subsample", 0.8, 1.0)
    #     }
    # },

    # "AdaBoost": {
    #     "model": AdaBoostRegressor(),
    #     "params": {
    #         "n_estimators": lambda trial: trial.suggest_int("n_estimators", 50, 200, step=50),
    #         "learning_rate": lambda trial: trial.suggest_float("learning_rate", 0.01, 1.0),
    #         "loss": lambda trial: trial.suggest_categorical("loss", ["linear", "square", "exponential"])
    #     }
    # },

    # "Hist Gradient Boosting": {
    #     "model": HistGradientBoostingRegressor(),
    #     "params": {
    #         "learning_rate": lambda trial: trial.suggest_float("learning_rate", 0.01, 0.2),
    #         "max_iter": lambda trial: trial.suggest_int("max_iter", 50, 200, step=50),
    #         "max_depth": lambda trial: trial.suggest_int("max_depth", 3, 7),
    #         "min_samples_leaf": lambda trial: trial.suggest_int("min_samples_leaf", 1, 5),
    #         "max_bins": lambda trial: trial.suggest_int("max_bins", 1, 255)
    #     }
    # },
}

# Objective Function for Optuna
def objective(trial, model_name, model_info, X_train, y_train, X_test, y_test):
    param_funcs = model_info["params"]
    params = {key: func(trial) for key, func in param_funcs.items()}  # Suggest parameters dynamically

    # Handle dynamic instantiation for CatBoost
    if model_name == "CatBoost":
        model = CatBoostRegressor(**params, verbose=0)
    else:
        model = model_info["model"].__class__(**params)

    model.fit(X_train, y_train)
    y_pred = np.floor(model.predict(X_test))  # Adjust predictions if needed

    mape = mean_absolute_percentage_error(y_test, y_pred)
    return mape

# Function to optimize models
def optimize_models(models_params, X_train, y_train, X_test, y_test, n_trials=50, save_path=None):
    best_models = {}
    optimized_models = {}  # To store the trained models

    for model_name, model_info in models_params.items():
        print(f"Optimizing {model_name}...")

        # Create study for each model
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, model_name, model_info, X_train, y_train, X_test, y_test), n_trials=n_trials)

        best_models[model_name] = {
            "best_params": study.best_trial.params,
            "best_value": study.best_trial.value
        }

        # Instantiate and train the best model with optimal parameters
        best_params = study.best_trial.params
        if model_name == "CatBoost":
            optimized_model = CatBoostRegressor(**best_params, verbose=0)
        else:
            optimized_model = model_info["model"].__class__(**best_params)

        optimized_model.fit(X_train, y_train)
        optimized_models[model_name] = optimized_model

        print(f"Best parameters for {model_name}: {study.best_trial.params}")
        logging.info(f"Best parameters for {model_name}: {study.best_trial.params}")
        print(f"Best MAPE: {study.best_trial.value}\n")
        logging.info(f"Best MAPE: {study.best_trial.value}\n")

    # Save the best_models dictionary using save_object
    save_object(save_path, best_models)

    # Print summary of results
    print("Summary of Best Models:")
    for model, result in best_models.items():
        print(f"{model}: MAPE={result['best_value']}, Params={result['best_params']}")

    return optimized_models, best_models

if __name__ == "__main__":
    save_path = os.path.join('artifacts', 'objects', "optuna.pkl")
    obj = DataIngestion()
    train_data, test_data, submission_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_df, test_df, _, _ = data_transformation.initiate_data_transformation(train_data, test_data, submission_data)

    X_train, y_train = train_df.drop(columns=["quantite_vendue"]), train_df["quantite_vendue"]
    X_test, y_test = test_df.drop(columns=["quantite_vendue"]), test_df["quantite_vendue"]

    # Run optimization and save the best models
    optimize_models(models_params, X_train, y_train, X_test, y_test, save_path=save_path)



