import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn .neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor

from src.exception_1 import CustomException
from src.logger_1 import logging
from src.utils_1 import  save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    
    def inititate_model_trainer(self, train_array,test_array,preprocessor_path):
        try:
            logging.info("Splitting training and test input data")
            x_train,y_train,x_test,y_test=(train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])
        
            models = {
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRFRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Linear Regression": LinearRegression(),
            }

            # REDUCED PARAMS - faster training for testing purposes
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse'],
                    'max_depth': [3, 5],
                },
                "Random Forest": {
                    'n_estimators': [8, 16],
                    'max_depth': [3, 5],
                },
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5],
                },
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [8, 16],
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [8, 16],
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8],
                    'learning_rate': [0.01, 0.1],
                    'iterations': [30, 50],
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [8, 16],
                },
                "Linear Regression": {},
            }

            model_report:dict=evaluate_models(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, models=models, param=params)
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
