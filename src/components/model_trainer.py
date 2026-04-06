import os
import sys
from dataclasses import dataclass
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor , GradientBoostingRegressor , RandomForestRegressor
from sklearn.metrics import r2_score 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model
from src.components.hyperParameter_tunning import hyperParameterTunning

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts" , "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_train_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self , train_array  , test_array ):
        try:
            logging.info("Split train test input data")
            
            # Extract math_score from the LAST column (added by data_transformation.py)
            y_train = train_array[:, -1]
            y_test = test_array[:, -1]
            
            # Get all columns except the last one (which is math_score)
            x_train = train_array[:, :-1]
            x_test = test_array[:, :-1]
            
            print(f"X_train shape: {x_train.shape}, Y_train shape: {y_train.shape}")
            print(f"X_test shape: {x_test.shape}, Y_test shape: {y_test.shape}")
            print(f"Y_train range: {y_train.min():.2f} - {y_train.max():.2f}")

            # Get best parameters from hyperparameter tuning
            logging.info("Getting best hyperparameters")
            hyperparameter_tuner = hyperParameterTunning()
            best_params = hyperparameter_tuner.hyperparameter(train_array, test_array)

            models = {
                # "Linear R" : LinearRegression(**best_params.get("liner", {})),
                "DecisionTree R" : DecisionTreeRegressor(**best_params.get("DecisionTree", {})),
                "GradientBoost R" : GradientBoostingRegressor(**best_params.get("GradientBoost", {})),
                "K-Neighbors R" : KNeighborsRegressor(**best_params.get("KNN", {})),
                "XGBoost R" : XGBRegressor(**best_params.get("xgboost", {})),
                "CatBoosting R" : CatBoostRegressor(verbose=False, **best_params.get("catBoost", {})),
                "Adaboost R" : AdaBoostRegressor(**best_params.get("Adaboost", {})),
                "randomForest" : RandomForestRegressor(**best_params.get("RF", {}))
            }

            model_report : dict = evaluate_model(x_train = x_train , y_train=y_train ,x_test=x_test, 
                                                y_test=y_test ,models=models)
            # to get the best model scoure from the dict
            best_model_scour = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_scour)]
            best_model = models[best_model_name]

            if best_model_scour < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both traning and testing")

            save_object(
                file_path=self.model_train_config.train_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            model_score = r2_score(y_test , predicted)
            return model_score
        except Exception as e:
            raise CustomException(e , sys)












