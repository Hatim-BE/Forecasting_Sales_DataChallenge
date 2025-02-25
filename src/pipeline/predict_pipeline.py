import sys
import os

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    # def predict(self, file=None):
    #     try:
    #         model_path=os.path.join("artifacts","models", "XGBoost Regressor.pkl")
                
    #         model = load_object(model_path)
    #         file = pd.read_csv("artifacts/transformed_data/transformed_submission.csv")
    #         predictions = np.floor(model.predict(file))
    #         return predictions
        
    #     except Exception as e:
    #         raise CustomException(e,sys)
        

    def predict(self, inputs=None):
        try:
            model_path=os.path.join("artifacts","models", "XGBoost Regressor.pkl")
            preprocessor_path=os.path.join('artifacts', 'preprocessor', 'preprocessor.pkl')  

            model = load_object(model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            data = preprocessor.transform(inputs)
            predictions = np.floor(model.predict(data))
            return predictions
        
        except Exception as e:
            raise CustomException(e,sys)
        


class CustomData:
    def __init__(
        self,
        Unnamed: 0,
        id_produit: str,
        date: str,
        categorie: str,
        marque: str,
        prix_unitaire: float,
        promotion: float,
        jour_ferie: str,
        weekend: str,
        stock_disponible: int,
        condition_meteo: str,
        region: str,
        moment_journee: str
    ):
        self.Unnamed = Unnamed
        self.id_produit = id_produit
        self.date = date
        self.categorie = categorie
        self.marque = marque
        self.prix_unitaire = prix_unitaire
        self.promotion = promotion
        self.jour_ferie = jour_ferie
        self.weekend = weekend
        self.stock_disponible = stock_disponible
        self.condition_meteo = condition_meteo
        self.region = region
        self.moment_journee = moment_journee

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Unnamed: 0": [0],
                "id_produit": [self.id_produit],
                "date": [self.date],
                "categorie": [self.categorie],
                "marque": [self.marque],
                "prix_unitaire": [self.prix_unitaire],
                "promotion": [self.promotion],
                "jour_ferie": [self.jour_ferie],
                "weekend": [self.weekend],
                "stock_disponible": [self.stock_disponible],
                "condition_meteo": [self.condition_meteo],
                "region": [self.region],
                "moment_journee": [self.moment_journee]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

