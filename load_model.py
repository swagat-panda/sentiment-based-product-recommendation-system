import pandas as pd
import numpy as np
import json

from joblib import load
import en_core_web_md
import pickle

model_path = './model/'


class Singleton(type):
    """
    Define an Instance operation that lets clients access its unique
    instance.
    """

    def __init__(cls, name, bases, attrs, **kwargs):
        super().__init__(name, bases, attrs)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class LoadModel(metaclass=Singleton):
    def __init__(self):
        self.vector_embedding = en_core_web_md.load()
        self.clf = load(model_path + 'svm_rbf.joblib')
        self.df_reco = pd.read_excel(model_path + 'df_reco.xlsx')
        with open(model_path + 'product_id_vocab.json') as f:
            self.product_id_vocab = json.load(f)
        with open(model_path + 'user_id_vocab.json') as f:
            self.user_id_vocab = json.load(f)
        with open(model_path + "user_final_rating.pkl", "rb") as f:
            self.user_final_rating = pickle.load(f)
