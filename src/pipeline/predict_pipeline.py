import sys
import os

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, file=None):
        try:
            model_path=os.path.join("artifacts","models", "XGBoost Regressor.pkl")
                
            model = load_object(model_path)
            file = pd.read_csv("artifacts/transformed_data/transformed_submission.csv")
            predictions = np.floor(model.predict(file))
            return predictions
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    predict_pipeline = PredictPipeline()
    predictions = predict_pipeline.predict()

