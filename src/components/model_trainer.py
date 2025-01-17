import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import *

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    
    def initiate_model_trainer(self,train_array=None,test_array=None):
        try:
            train_array = pd.read_csv("artifacts/transformed_data/transformed_train.csv")
            test_array = pd.read_csv("artifacts/transformed_data/transformed_test.csv")
            X_train, y_train = train_array.drop("quantite_vendue", axis=1), train_array["quantite_vendue"]

            best_models = load_object("artifacts/objects/optuna.pkl")

            for model_name, model_info in best_models.items():
                print(f"Retraining {model_name} with best parameters...")

                best_params = model_info["best_params"]

                if model_name == "XGBoost Regressor":
                    model = XGBRegressor(**best_params)
                elif model_name == "CatBoost":
                    model = CatBoostRegressor(**best_params, verbose=0)
                elif model_name == "Random Forest Regressor":
                    model = RandomForestRegressor(**best_params)
                else:
                    raise ValueError(f"Model {model_name} not supported")

                # Train the model
                model.fit(X_train, y_train)

                model_save_path = os.path.join('artifacts', 'models', f"{model_name}.pkl")
                save_object(model_save_path, model)
                print(f"Saved trained model: {model_name} at {model_save_path}")

            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer()