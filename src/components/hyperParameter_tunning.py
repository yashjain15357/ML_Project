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
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging

from src.utils import evaluate_model

class hyperParameterTunning:
    def __init__(self):
        pass

        
    def hyperparameter(self , train_array  , test_array):
        try:
            logging.info("Split train test input data")
            x_train , y_train  , x_test , y_test = (
                train_array[:,:-1],
                train_array[: ,-1],
                test_array[: , : -1],
                test_array[: , -1]
            )
            parameter = {


                # 'linear_param' : {
                #     'fit_intercept': [True, False]
                # },
                'decisionTree_param' : {
                    'criterion':['squared_error', 'friedmad', 'absolute_error'],
                    'splitter':['best', 'random'],
                    'max_depth':[1,2,3,4,5],
                    'max_features':['sqrt', 'log2']
                } ,
                'gradientBoost_param' : {
                    'loss' : ['squared_error', 'huber', 'absolute_error'],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200, 500],
                    'criterion': ['friedman_mse', 'squared_error'],
                    'max_depth': [3, 5, 7]
                },

                'knn_params' : {"n_neighbors": [2, 3, 10, 20, 40, 50]},
                # XGBoost parameters for regression
                'xgboost_param' : {
                    'n_estimators': [50, 100, 200, 500],
                    'max_depth': [3, 5, 7, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                },
                'adaboost_param' : {
                    'n_estimators' : [50,60,70,80,90],
                    'loss' : ['linear', 'square', 'exponential']
                },
                'catboost_param' : {
                    'iterations': [100, 200, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [4, 6, 8, 10],
                    'l2_leaf_reg': [1, 3, 5, 7, 9],
                    'border_count': [32, 64, 128, 255]
                },
                'randomforest_params' : {
                    "max_depth": [5, 8, 15, None, 10],
                    "max_features": [5, 7, "auto", 8],
                    "min_samples_split": [2, 8, 15, 20],
                    "n_estimators": [100, 200, 500, 1000]}
            }

            randomcv_models = [
                # ("liner" , LinearRegression() , parameter["linear_param"]),
                ("DecisionTree" , DecisionTreeRegressor() , parameter["decisionTree_param"]),
                ("GradientBoost" , GradientBoostingRegressor() , parameter['gradientBoost_param']),
                ("KNN" , KNeighborsRegressor() , parameter['knn_params']),
                ("xgboost" , XGBRegressor() , parameter['xgboost_param']),
                ("Adaboost" , AdaBoostRegressor() , parameter['adaboost_param']),
                ("RF", RandomForestRegressor(), parameter['randomforest_params']),
                ("catBoost" , CatBoostRegressor() , parameter["catboost_param"])  
                ]
            model_param = {}
            print("process of find best parameter is start")
            for name , model, param in randomcv_models :
                randomCV = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param,
                    n_iter=100,
                    verbose=2,
                    n_jobs=-1)
                randomCV.fit(x_train , y_train)
                model_param[name] = randomCV.best_params_
            print("process of find best parameter is end")
            return model_param


        except Exception as e:
            raise CustomException(e , sys)

    