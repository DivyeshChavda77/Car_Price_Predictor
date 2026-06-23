import os
import sys

import pandas as pd
import numpy as np

from dataclasses import dataclass

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            prediction_log = model.predict(data_scaled)
            prediction = np.expm1(prediction_log)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)
        
@dataclass
class CustomData:
    km_driven: float
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    brand: str
    car_age: int

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                
                "km_driven": [self.km_driven],

                "fuel": [self.fuel],

                "seller_type": [self.seller_type],

                "transmission": [self.transmission],

                "owner": [self.owner],

                "brand": [self.brand],

                "car_age": [self.car_age]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)