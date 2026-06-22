import os
import sys
import numpy as np

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):

        try:
            logging.info("Splitting training and testing arrays")

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            # LOG TRANSFORM TARGET
            y_train = np.log1p(y_train)
            y_test = np.log1p(y_test)

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(random_state=42),
                "XGBoost": XGBRegressor(random_state=42),

                "CatBoost": CatBoostRegressor(verbose=False,random_state=42)
            }

            model_report = evaluate_models(X_train,y_train,X_test,y_test,models)

            print("\nModel Scores\n")
            print(model_report)

            best_model_score = max(model_report.values())

            best_model_name = max(model_report,key=model_report.get)

            best_model = models[best_model_name]

            print(f"\nBest Model : {best_model_name}")
            print(f"Best Score : {best_model_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)

            logging.info("Model Training Completed")

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)