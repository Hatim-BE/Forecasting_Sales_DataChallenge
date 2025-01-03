import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from src.logger import logging
import sys
from dataclasses import dataclass
from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging
from utils.functions import *
from src.utils import save_object
import warnings
warnings.filterwarnings("ignore")
# Custom Transformers for specific logic
class ReplaceMarque(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        marque_to_product = X_copy.dropna(subset=['id_produit', 'marque']).groupby('id_produit')['marque'].unique().to_dict()
        X_copy['marque'] = X_copy.apply(lambda row: self.replace_marque(row, marque_to_product), axis=1)
        X_copy['marque'] = X_copy.groupby(
        X_copy['id_produit'].fillna(X_copy['categorie'])
        )['marque'].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown')
        )
        return X_copy
    
    def replace_marque(self, row, marque_to_product):
        if pd.isna(row['marque']) and row['id_produit'] in marque_to_product:
            return marque_to_product[row['id_produit']][0]
        return row['marque']
class FillWeekend(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy['date'] = pd.to_datetime(X_copy['date'])  # Ensure 'date' is in datetime format
        X_copy['weekend'] = X_copy['date'].dt.weekday >= 5
        return X_copy
    
class FillPrix(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy['prix_unitaire'] = X_copy.groupby(
        X_copy['id_produit'].fillna(X_copy['categorie'])
        )['prix_unitaire'].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.mean())
        )
        return X_copy
class FillStock(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy['stock_disponible'] = X_copy.groupby(
        X_copy['id_produit'].fillna(X_copy['categorie'])
        )['stock_disponible'].transform(
            lambda x: x.fillna(x.mean())
        )
        return X_copy
    
class AddDateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy['date'] = pd.to_datetime(X_copy['date'])  # Ensure 'date' is in datetime format
        X_copy["month"] = X_copy["date"].dt.month
        X_copy["day"] = X_copy["date"].dt.day
        X_copy["quarter"] = X_copy["date"].dt.quarter
        return X_copy
class LogTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy['prix_unitaire'] = np.log1p(X_copy['prix_unitaire'])
        X_copy['stock_disponible'] = np.log1p(X_copy['stock_disponible'])
        return X_copy
class CustomColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
        
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        return X_copy.drop(columns=self.columns_to_drop)

class CustomNullValuesDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        return X_copy.dropna(subset=self.columns)
    
# Custom Imputation for "jour_ferie"
class ImputeJourFerie(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy['date'] = pd.to_datetime(X_copy['date'])  # Ensure 'date' is in datetime format
        # Step 1: Extract unique (day, month) -> jour_ferie mapping
        mapping_df = X_copy.dropna(subset=['jour_ferie'])[X_copy["jour_ferie"] == 1]  # Only use rows where 'jour_ferie' is not NaN
        jour_ferie_map = dict(zip(
            zip(mapping_df['date'].dt.day, mapping_df['date'].dt.month),
            mapping_df['jour_ferie']
        ))
        X_copy['jour_ferie'] = X_copy.apply(
            lambda row: jour_ferie_map.get((row['date'].day, row['date'].month), 0),
            axis=1
        )
        return X_copy
# Custom Imputation for "promotion"
class ImputePromotion(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        promo = X.copy()
        # Ensure 'date' is in datetime format
        promo['date'] = pd.to_datetime(promo['date'], errors='coerce')
        # Step 1: Add 'day' and 'month' columns
        promo['day'] = promo['date'].dt.day
        promo['month'] = promo['date'].dt.month
        # Step 2: Extract the mapping for (day, month, jour_ferie) -> promotion
        promo_mapping = promo.dropna(subset=['promotion']).groupby(['day', 'month', 'jour_ferie'])['promotion'].max()
        promo_mapping = promo_mapping.to_dict()
        # Step 3: Impute missing 'promotion' using the mapping
        promo['promotion'] = promo.apply(
            lambda row: promo_mapping.get((row['day'], row['month'], row['jour_ferie']), 0)
            if pd.isna(row['promotion']) else row['promotion'],
            axis=1
        )
        return promo
class RandomFillCategorical(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    def fit(self, X, y=None):
        # No fitting is required for this transformation, so return self
        return self
    def transform(self, X):
        np.random.seed(42)
        for col in self.columns:
            # Only fill missing values (NaN) with a random choice from existing values
            X[col] = X[col].fillna(np.random.choice(X[col].dropna().unique()))
        return X
class WinsorizeColumn(BaseEstimator, TransformerMixin):
    def __init__(self, columns, upper=85, lower=15):
        self.columns = columns
        self.upper = upper
        self.lower = lower
    def fit(self, X, y=None):
        return self  # No fitting necessary
    def transform(self, X):
        X_copy = X.copy()  # Make a copy of the dataframe to avoid modifying the original data
        # Apply winsorization on the column data
        X_copy = winsorize(X_copy, self.columns, upper=self.upper, lower=self.lower)
        return X_copy
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor', "preprocessor.pkl")
    transf_train_data_path: str = os.path.join('artifacts', 'transformed_data', 'transformed_train.csv')
    transf_test_data_path: str = os.path.join('artifacts', 'transformed_data', 'transformed_test.csv')
    transf_submission_data_path: str = os.path.join('artifacts', 'transformed_data', 'transformed_submission.csv')
    
# Custom DataTransformer class to integrate all transformations
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_object(self, column_names):
        '''
        Returns a ColumnTransformer with a series of custom transformations for the dataset.
        '''
        try:
            # Define the overall custom pipeline including domain-specific logic
            custom_pipeline = Pipeline(
                steps=[
                    ("fill_weekend", FillWeekend()),
                    ("replace_marque", ReplaceMarque()),
                    ("add_date_features", AddDateFeatures()),
                    ("impute_jour_ferie", ImputeJourFerie()),
                    ("impute_promotion", ImputePromotion()),
                    ("fill_prix", FillPrix()),
                    ("fill_stock", FillStock()),
                    ("log_transform", LogTransform()),
                    ("random_fill_categorical", RandomFillCategorical(columns=['condition_meteo', 'moment_journee', 'region'])),
                    ("drop_columns", CustomColumnDropper(columns_to_drop=["date", "Unnamed: 0"])),
                    ("winsorize", WinsorizeColumn(columns=["prix_unitaire", "stock_disponible"], upper=85, lower=15)),
                ]
            )
            # Integrate both the standard preprocessing and custom logic
            preprocessor = ColumnTransformer(
                [
                    ("custom_transform", custom_pipeline, column_names)
                ]
            )
            return preprocessor
        
        except Exception as e:
            logging.error(f"Error in creating data transformer object: {e}")
            raise CustomException(e, sys)
    def initiate_data_transformation(self, train_path, test_path, submission_path):
        try:
            target = "quantite_vendue"
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            submission_df=pd.read_csv(submission_path)

            target_column = train_df['quantite_vendue']
            ['Unnamed: 0', 'id_produit', 'date', 'categorie', 'marque', 'prix_unitaire', 'promotion', 'jour_ferie', 'weekend', 'stock_disponible', 'condition_meteo', 'region', 'moment_journee', 'month', 'day', 'quarter']
            ['Unnamed: 0', 'id_produit', 'date', 'categorie', 'marque', 'prix_unitaire', 'promotion', 'jour_ferie', 'weekend', 'stock_disponible', 'condition_meteo', 'region', 'moment_journee']
            # Remove the target column from the original data for transformation
            train_df_without_target = train_df.drop(columns=['quantite_vendue'])
            test_df_without_target = test_df.drop(columns=['quantite_vendue'])

            # Define your transformed columns (without the target column for now)
            tranformed_columns = list(train_df_without_target.drop(["date", "Unnamed: 0"], axis=1).columns) + ["month", "day", "quarter"]            

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object(train_df_without_target.columns)
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            
            train_arr = preprocessing_obj.fit_transform(train_df_without_target)
            test_arr = preprocessing_obj.transform(test_df_without_target)            
            submission_arr = preprocessing_obj.transform(submission_df)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info(f"Saved preprocessing object.")
            os.makedirs(os.path.dirname(self.data_transformation_config.transf_train_data_path), exist_ok=True)

            train_transformed_df = pd.DataFrame(train_arr, columns=tranformed_columns)
            train_transformed_df[target] = train_df[target]
            train_transformed_df = drop_null_values_columns(train_transformed_df, ["id_produit", "quantite_vendue"])

            test_transformed_df = pd.DataFrame(test_arr, columns=tranformed_columns)
            test_transformed_df[target] = test_df[target]
            test_transformed_df = drop_null_values_columns(train_transformed_df, ["id_produit", "quantite_vendue"])
            
            submission_transformed_df = pd.DataFrame(submission_arr, columns=tranformed_columns)

            X_train, y_train = train_transformed_df.drop(columns=["quantite_vendue"]), train_transformed_df["quantite_vendue"]
            X_test, y_test = test_transformed_df.drop(columns=["quantite_vendue"]), test_transformed_df["quantite_vendue"]


            # Call the encoding function
            X_train, X_test, submission_transformed_df = encode_features(
                X_train, y_train, X_test, submission_transformed_df
            )
            
            # Convert to float
            X_train, X_test, submission_transformed_df = convert_object_to_float(X_train, X_test, submission_transformed_df)
            
            train_transformed_df = pd.concat([X_train, y_train], axis=1)
            test_transformed_df = pd.concat([X_test, y_test], axis=1)
            
            train_transformed_df.to_csv(self.data_transformation_config.transf_train_data_path, index=False, header=True)
            test_transformed_df.to_csv(self.data_transformation_config.transf_test_data_path, index=False, header=True)
            submission_transformed_df.to_csv(self.data_transformation_config.transf_submission_data_path, index=False, header=True)
            logging.info(f"Saved tranformed data.")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.error(f"{e}")
            raise CustomException(e,sys)
if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data, submission_data=obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data, submission_data)