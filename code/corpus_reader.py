import os
import re
import pandas as pd

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


class CorpusReader:

    @staticmethod
    def get_question_pair_data(data_path_train, data_path_test, dev_size=0.05, random_state=42):

        df_train = pd.read_pickle(data_path_train)
        df_train = df_train[['q1_sent_idx', 'q2_sent_idx', 'is_duplicate']]
        df_train["sentence_pairs"] = df_train["q1_sent_idx"]
        df_train.apply(lambda x: x.sentence_pairs.extend(x.q2_sent_idx), axis=1)

        X_train = df_train["sentence_pairs"].tolist()
        Y_train = df_train['is_duplicate'].tolist()

        X_train, X_dev, Y_train, Y_dev = train_test_split(
            X_train, Y_train, test_size=dev_size, random_state=random_state
        )

        # df_test = pd.read_pickle(data_path_test)
        # df_test = df_test[['q1_sent_idx', 'q2_sent_idx']]
        # df_test["sentence_pairs"] = df_test["q1_sent_idx"]
        # df_test.apply(lambda x: x.sentence_pairs.extend(x.q2_sent_idx), axis=1)
        #
        # X_test = df_test["sentence_pairs"].tolist()
        X_test = list()
        Y_test = list()

        return X_train, Y_train, X_dev, Y_dev, X_test, Y_test
