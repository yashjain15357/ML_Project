import sys
import os
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class data_transformation_Config:
    # """
    # Configuration class for data transformation.
    
    # Purpose: Stores the file path where the preprocessor object will be saved after fitting
    
    # Attributes:
    #     prepocessor_obj_file_path (str): Path to save the fitted preprocessor pickle file
    
    # Usage: Used by DataTransformation class to manage preprocessor storage location
    # """
    prepocessor_obj_file_path = os.path.join('artifacts' , 'preprocessor.pkl')

class DataTransformation:
    # """
    # Data Transformation component for ML pipeline.
    
    # Purpose: Handles data preprocessing including:
    #     - Handling missing values (imputation)
    #     - Encoding categorical variables
    #     - Scaling numerical features
    
    # Workflow:
    #     1. Gets the preprocessor object with all transformation pipelines
    #     2. Applies preprocessing to training and test data
    #     3. Saves the fitted preprocessor for later use in prediction
    
    # Usage:
    #     obj = DataTransformation()
    #     train_array, test_array, preprocessor_path = obj.initiate_data_transformation(train_path, test_path)
    # """
    def __init__(self):
        # """
        # Initialize DataTransformation with configuration.
        
        # Purpose: Sets up the transformation configuration that defines preprocessor storage path
        # """
        self.data_transformation_config = data_transformation_Config()

    def get_data_transformer_object(self):
        # """
        # Create and return the preprocessor object with all transformation pipelines.
        
        # Purpose: Builds the complete preprocessing pipeline for both numerical and categorical features
        
        # Process:
        #     1. Numerical Pipeline:
        #        - Imputes missing values with median
        #        - Scales features using StandardScaler
            
        #     2. Categorical Pipeline:
        #        - Imputes missing values with most frequent value
        #        - One-hot encodes categorical variables
        #        - Scales the encoded features
            
        #     3. Combines both pipelines using ColumnTransformer
        
        # Returns:
        #     ColumnTransformer: Fitted preprocessor object ready to transform data
        
        # Raises:
        #     CustomException: If any error occurs during preprocessor creation
        # """
        try:
            int_feature= ["writing_score", "reading_score"]
            cat_feature = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            # Pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="median")),
                    ('scaler', StandardScaler())
                ]
            )
            logging.info("Numerical columns pipeline created")
            
            # Pipeline for categorical features
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ('onehot_encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
                    ("scaler", StandardScaler())
                ] 
            )
            logging.info("Categorical columns pipeline created")

            # Combine both pipelines
            # 👉 Ye kya karta hai:

            # Numerical columns → num_pipeline apply
            # Categorical columns → cat_pipeline apply
            # Dono ko combine karke final transformed dataset deta hai
            Preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, int_feature),
                    ("cat_pipeline", cat_pipeline, cat_feature)
                ]
            )

            return Preprocessor
        except Exception as e:
            raise CustomException( e , sys)
    
    def initate_data_transformation(self, train_path, test_path):
        # """
        # Execute the data transformation process.
        
        # Purpose: Main method that orchestrates preprocessing of training and test data
        
        # Steps:
        #     1. Load training and test CSV files
        #     2. Get the preprocessor object with all transformation pipelines
        #     3. Separate features (X) and target (y) from training data
        #     4. Apply preprocessing to training data (fit_transform)
        #     5. Apply preprocessing to test data (transform only)
        #     6. Combine transformed features with target variables
        #     7. Save the fitted preprocessor object for future use
        
        # Parameters:
        #     train_path (str): Path to training CSV file
        #     test_path (str): Path to test CSV file
        
        # Returns:
        #     tuple: (train_array, test_array, preprocessor_path)
        #            - train_array: Preprocessed training features and target
        #            - test_array: Preprocessed test features and target
        #            - preprocessor_path: Path where preprocessor was saved
        
        # Raises:
        #     CustomException: If any error occurs during transformation
        # """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("read train and test data completed")

            logging.info("obtaining Preprocessing object")
            Preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns='math_score')
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns='math_score')
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=Preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=Preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
            

            save_object(
                file_path=self.data_transformation_config.prepocessor_obj_file_path,
                obj = Preprocessor_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.prepocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)