import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngessionConfig:
    """
    Configuration class for data ingestion paths.
    
    Purpose: Stores the file paths where raw, training, and test data will be saved.
    
    Attributes:
        train_data_path (str): Path to save the training dataset after train-test split
        testdata_path (str): Path to save the test dataset after train-test split
        raw_data_path (str): Path to save the raw dataset without any split
    
    Usage: Used by DataIngestion class to manage output file locations
    """
    train_data_path: str = os.path.join("artifacts" , 'train.csv')
    testdata_path: str = os.path.join("artifacts" , 'test.csv')
    raw_data_path: str = os.path.join("artifacts" , 'data.csv')

class DataIngestion :
    """
    Data Ingestion component for ML pipeline.
    
    Purpose: Handles loading raw data, splitting it into training and test sets,
    and saving them to specified artifact paths.
    
    Workflow:
        1. Loads raw data from source CSV file
        2. Saves raw data to artifacts folder
        3. Performs 80-20 train-test split (80% train, 20% test)
        4. Saves split datasets to separate files
        5. Returns paths to both training and test datasets
    
    Usage:
        obj = DataIngestion()
        train_path, test_path = obj.initiate_data_ingestion()
    """
    def __init__(self):
        """
        Initialize DataIngestion with configuration.
        
        Purpose: Sets up the ingestion configuration that defines output paths
        """
        self.ingestion_config = DataIngessionConfig()

    def initiate_data_ingestion(self):
        """
        Execute the data ingestion process.
        
        Purpose: Main method that orchestrates the entire data loading and splitting process
        
        Steps:
            1. Read raw data from notebook/data/stud_eda.csv
            2. Create artifacts directory if it doesn't exist
            3. Save the raw data as-is to artifacts/data.csv
            4. Split data into 80% train and 20% test (random_state=42 for reproducibility)
            5. Save training data to artifacts/train.csv
            6. Save test data to artifacts/test.csv
            7. Log all operations for monitoring
        
        Returns:
            tuple: (train_data_path, test_data_path) - Paths to the split datasets
        
        Raises:
            CustomException: If any error occurs during ingestion
        """
        logging.info("Enter data integration method or component")
        try:
            df = pd.read_csv("notebook/data/stud_eda.csv")
            logging.info('read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path ),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path , index=False , header=True)
            logging.info("Train test split initiated")

            train_set , test_set = train_test_split(df , test_size=0.2  ,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path , index= False , header = True)
            test_set.to_csv(self.ingestion_config.testdata_path , index = False , header = True)

            logging.info("Inmgestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.testdata_path
            )
        except Exception as e:
            raise CustomException(e , sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()