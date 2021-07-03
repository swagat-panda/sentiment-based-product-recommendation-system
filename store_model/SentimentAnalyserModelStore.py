'''
As per the Analysis done on the notebook,
i m writing the code to train and store the with the best parameters of SVM_RBF from GridSearchCV
'''

import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from bs4 import BeautifulSoup

import spacy
from spacy.tokens.doc import Doc
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import en_core_web_md
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from joblib import dump, load

# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load('en_core_web_md')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()
vector_embedding = en_core_web_md.load()
MAX_CV_FOLDS = 5


class TrainModel(object):
    def __init__(self):
        self.df = pd.read_csv('../data/sample30.csv')
        self.df = self.df[['reviews_text', 'reviews_title', 'user_sentiment']]
        self.df = self.df.dropna()
        convert = {"Positive": 1, "Negative": 0}
        self.df.user_sentiment = self.df.user_sentiment.apply(lambda x: convert[x])
        self.df['review'] = self.df.apply(lambda x: x['reviews_text'] + " " + x['reviews_title'], axis=1)
        self.tuned_parameters = [{"C": [
            1

        ],
            "gamma": [

                0.1
            ],
            "kernel": ["rbf"]

        }]

    def clean_review(self, review):
        review = BeautifulSoup(review, "lxml").get_text()
        # Removing the @
        review = re.sub(r"@[A-Za-z0-9]+", ' ', review)
        # Removing the URL links
        review = re.sub(r"https?://[A-Za-z0-9./]+", ' ', review)
        # Keeping only letters
        review = re.sub(r"[^a-zA-Z.!?']", ' ', review)
        # Removing additional whitespaces
        review = re.sub(r" +", ' ', review)
        return review

    def get_vector(self, sentence):
        doc = vector_embedding(sentence)
        return np.asarray(doc.vector)

    def spacy_tokenizer(self, sentence):
        # Creating our token object, which is used to create documents with linguistic annotations.
        mytokens = parser(sentence)

        # Lemmatizing each token and converting each token into lowercase
        mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]

        # Removing stop words
        mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]

        # return preprocessed list of tokens
        return mytokens

    def preprocess(self):
        self.df.review = [self.clean_review(review) for review in tqdm(self.df.review)]
        df_req = self.df[['review', 'user_sentiment']]
        tokenised_output = df_req.review.apply(self.spacy_tokenizer)
        vector_output = []
        for sentence in tqdm(tokenised_output):
            vector_output.append(self.get_vector(' '.join(sentence)))

        vector_output = np.asarray(vector_output)
        USER_SENTIMENT = df_req.user_sentiment

        X_train, X_test, y_train, y_test = train_test_split(vector_output, USER_SENTIMENT, test_size=0.3)

        smt = SMOTE()

        x_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
        return x_train_sm, y_train_sm, X_test, y_test

    def train_model(self, x_train_sm, y_train_sm, X_test, y_test):
        # x_train_sm, y_train_sm, X_test, y_test = self.preprocess()
        cv_splits = max(2, min(MAX_CV_FOLDS, np.min(np.bincount(y_train_sm)) // 5))
        clf_svm = GridSearchCV(SVC(C=1, probability=True, class_weight='balanced'),
                               param_grid=self.tuned_parameters, n_jobs=-1, return_train_score=True,
                               cv=cv_splits, scoring='accuracy', verbose=2)
        clf_svm.fit(x_train_sm, y_train_sm)
        print("Train Score: ", clf_svm.best_score_)
        svm_model = clf_svm.best_estimator_
        return svm_model

    def persist(self, model):
        dump(model, "../model/svm_rbf.joblib")
        return True

    def process(self):
        x_train_sm, y_train_sm, X_test, y_test = self.preprocess()
        model = self.train_model(x_train_sm, y_train_sm, X_test, y_test)
        self.persist(model)
        ## Let's Evaluate this Model
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        return True


if __name__ == '__main__':
    obj = TrainModel()
    print(obj.process())
