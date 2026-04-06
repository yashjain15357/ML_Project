import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object


class predict_pipeline:
    def __inti__(self):
        pass


    def predict(self, feature):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Ensure feature is a DataFrame
            if not isinstance(feature, pd.DataFrame):
                feature = pd.DataFrame(feature)
            
            print(f"Feature shape before preprocessing: {feature.shape}")
            print(f"Feature columns: {feature.columns.tolist()}")
            
            data_scaled = preprocessor.transform(feature)
            print(f"Data shape after preprocessing: {data_scaled.shape}")
            
            preds = model.predict(data_scaled)
            
            return preds
        except Exception as e:
            raise CustomException(e, sys)
        
    

class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            # Order must match the training data columns (excluding math_score which is the target)
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            df = pd.DataFrame(custom_data_input_dict)
            print(f"Input DataFrame shape: {df.shape}")
            print(f"Input DataFrame columns: {df.columns.tolist()}")
            return df

        except Exception as e:
            raise CustomException(e, sys)

