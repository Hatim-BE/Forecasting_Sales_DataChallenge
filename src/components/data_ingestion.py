import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

sub_path = "https://raw.githubusercontent.com/dataafriquehub/donnee_vente/refs/heads/main/submission.csv"
train_path = "https://raw.githubusercontent.com/dataafriquehub/donnee_vente/refs/heads/main/train.csv"

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")
    sub_data_path: str = os.path.join('artifacts', "submission.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(train_path)
            sub = pd.read_csv(sub_path)

            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Separate rows with and without missing target values
            train_with_target = df[df['quantite_vendue'].notna()]  # Rows with no missing target
            train_missing_target = df[df['quantite_vendue'].isna()]  # Rows with missing target

            # Split the data with no missing target into training and test sets
            X = train_with_target.drop(['quantite_vendue'], axis=1)  # Exclude the target variable
            y = train_with_target['quantite_vendue']

            # Perform the train-test split with no shuffle
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=False)

            # Add rows with missing target back to the training set
            X_train_final = pd.concat([X_train, train_missing_target.drop(['quantite_vendue'], axis=1)])
            y_train_final = pd.concat([y_train, train_missing_target['quantite_vendue']])

            # Prepare final test data
            X_test_final = X_test.copy()
            y_test_final = y_test.copy()

            # Print shapes to verify
            print(f"Training data shape: X_train={X_train_final.shape}, y_train={y_train_final.shape}")
            print(f"Test data shape: X_test={X_test_final.shape}, y_test={y_test_final.shape}")

            # Save the final data to CSV
            pd.concat([X_train_final, y_train_final], axis=1).to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            pd.concat([X_test_final, y_test_final], axis=1).to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            sub.to_csv(self.ingestion_config.sub_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path, self.ingestion_config.sub_data_path)

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data, aubmiaaion_data = obj.initiate_data_ingestion()
