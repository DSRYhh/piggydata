import numpy as np
import pandas as pd
from sklearn import linear_model

import Checker


def split_data(data: pd.DataFrame):
    """
    Split a DataFrame to 3 parts:
    Training set        60%
    Validation set		20%
    Test set	    	20%
    :param data:
    :return a tuple (training_set, validation_set, test_set)
    each set is a pandas DataFrame
    """
    data = data.sample(frac=1)

    data_train = data[:int(len(data) * 0.6)]
    data_validation = data[int(len(data) * 0.6 + 1): int(len(data) * 0.8)]
    data_test = data[int(len(data) * 0.8 + 1):]

    return data_train, data_validation, data_test


def train(x, y):
    """
    Train linear regression model for input x and y
    :param x: feature matrix with columns: 日期, 生猪存栏, 母猪存栏
    :param y: the price of each data
    :return: the linear mapping matrix from features to price
    """

    # TODO YOUR CODE HERE
    # Create a linear regression model and train it using x and y, then return coefficient matrix
    pass


def normalize(x):
    """
    Scale all input features to the range [0, 1].
    :param x: input features
    :return: tuple: (a numpy numerical matrix with normalized features inside, price)
    """
    date = x['日期']
    pig_num = x['生猪存栏（万头）']
    sow_num = x['母猪存栏（万头）']
    price = x['苗猪平均价格（元/公斤）']

    # transform date to a int or float list or numpy array
    # hint: use timestamp() method of pandas.Timestamp to convert date to int
    # TODO YOUR CODE HERE

    # normalize pig_num, date, sow_num
    # TODO YOUR CODE HERE

    return np.hstack((date, pig_num, sow_num)), price


if __name__ == '__main__':
    Checker.normalization_check()
