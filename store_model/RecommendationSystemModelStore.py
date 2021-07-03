'''
As per the Analysis done on the notebook,
i m going ahead with User based recommendation
'''
import json

import numpy as np
import warnings
import pandas as pd
from tqdm import tqdm
import re
import pickle

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from numpy import *
from sklearn.model_selection import train_test_split


class RecommendationSystem(object):
    def __init__(self):
        self.df = pd.read_csv('../data/sample30.csv')
        self.df_reco = self.df[["categories", "name", "reviews_rating", "reviews_username", "reviews_text"]]
        self.df_reco = self.df_reco.dropna()
        self.df_reco = self.df_reco.drop_duplicates()

    def create_vocab(self):
        user_id_vocab = {}
        product_id_vocab = {}

        user_name = list(set(list(self.df_reco.reviews_username)))
        product_name = list(set(list(self.df_reco.name)))
        counter_user_id = 0
        counter_product_id = 0

        for user in user_name:
            user_id_vocab[user] = counter_user_id
            counter_user_id = counter_user_id + 1

        for product in product_name:
            product_id_vocab[product] = counter_product_id
            counter_product_id = counter_product_id + 1

        self.df_reco['user_id'] = self.df_reco.reviews_username.apply(lambda x: user_id_vocab[x])
        self.df_reco['product_id'] = self.df_reco.name.apply(lambda x: product_id_vocab[x])
        return user_id_vocab, product_id_vocab

    def persist(self, user_id_vocab, product_id_vocab, user_final_rating):
        self.df_reco.to_excel("../model/df_reco.xlsx")
        with open("../model/user_id_vocab.json", "w") as f:
            json.dump(user_id_vocab, f, indent=4)

        with open("../model/product_id_vocab.json", "w") as f:
            json.dump(product_id_vocab, f, indent=4)

        with open("../model/user_final_rating.pkl", "wb") as f:
            pickle.dump(user_final_rating, f, protocol=pickle.HIGHEST_PROTOCOL)

        return True

    def process_model(self):
        user_id_vocab, product_id_vocab = self.create_vocab()
        train, test = train_test_split(self.df_reco, test_size=0.30, random_state=31)
        df_pivot_train = train.pivot_table(
            index='user_id',
            columns='product_id',
            values='reviews_rating'
        ).fillna(0)
        dummy_train = train.copy()
        dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x >= 1 else 1)
        dummy_train = dummy_train.pivot_table(
            index='user_id',
            columns='product_id',
            values='reviews_rating'
        ).fillna(1)
        df_pivot_adj = train.pivot_table(
            index='user_id',
            columns='product_id',
            values='reviews_rating'
        )
        # Normalising the rating of the product for each user around 0 mean
        mean = np.nanmean(df_pivot_adj, axis=1)
        df_subtracted = (df_pivot_adj.T - mean).T
        # Creating the User Similarity Matrix using pairwise_distance function.
        user_correlation_adj = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
        user_correlation_adj[np.isnan(user_correlation_adj)] = 0

        user_correlation_adj[user_correlation_adj < 0] = 0
        user_predicted_ratings = np.dot(user_correlation_adj, df_pivot_adj.fillna(0))
        user_final_rating = np.multiply(user_predicted_ratings, dummy_train)

        self.persist(user_id_vocab, product_id_vocab, user_final_rating)

        return True


if __name__ == '__main__':
    obj = RecommendationSystem()
    print(obj.process_model())
