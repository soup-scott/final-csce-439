from defender.features import PEFeatureExtractor
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import gzip, pickle
import logging
import json
from defender.loader18 import parse_one_sample


logging.basicConfig(level=logging.DEBUG)


class RFModel(object):
    '''Implements predict(self, bytez)'''
    def __init__(self, model_gz_path: str, name: str, model_thresh: float):
        # load model
        with gzip.open(model_gz_path, "rb") as f:
            saved = pickle.load(f)
        
        self.model = saved["model"]
        self.feature_names = saved["features"]
        self.model_gz_path = model_gz_path
        self.model_thresh = model_thresh
        self.__name__ = name
        self.extractor = PEFeatureExtractor()

        print("Model Loaded...\nReady for querying...\n")

    def extract_selected_features(self, bytez):

        # Get raw features (dictionary)
        s = self.extractor.raw_features(bytez)

        row = parse_one_sample(s)

        # Combine into a feature vector

        return row

    def predict(self, bytez: bytes) -> int:

        self.features = self.extract_selected_features(bytez)


        # X_new = pd.DataFrame([self.features], columns=self.feature_names)  # wrap in list for single sample
        X_new = self.features.reshape(1, -1)

        # print(X_new)
        result = self.model.predict_proba(X_new)[0, 1]
        return int(result > self.model_thresh)

    def predict_proba(self, bytez: bytes) -> float:

        self.features = self.extract_selected_features(bytez)


        # X_new = pd.DataFrame([self.features], columns=self.feature_names)  # wrap in list for single sample
        X_new = self.features.reshape(1, -1)

        # print(X_new)
        result = self.model.predict_proba(X_new)[0, 1]
        return result

    def model_info(self) -> dict:
        return {"model_gz_path": self.model_gz_path,
                "name": self.__name__}


    
