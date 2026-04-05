import os
import sys
from dataclasses import dataclass

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

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts" , "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_train_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self , train_array  , test_array ):
        try:
            logging.info("Split train test input data")
            x_train , y_train  , x_test , y_test = (
                train_array[:,:-1],
                train_array[: ,-1],
                test_array[: , : -1],
                test_array[: , -1]
            )

            models = {
                "Linear R" : LinearRegression(),
                "DecisionTree R" : DecisionTreeRegressor(),
                "GradientBoost R" : GradientBoostingRegressor(),
                "K-Neighbors R" : KNeighborsRegressor(),
                "XGBoost R" : XGBRegressor(),
                "CatBoosting R" : CatBoostRegressor(verbose=False),
                "Adaboost R" : AdaBoostRegressor()
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












