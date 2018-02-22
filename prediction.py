import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

import Checker
from dataReader import read_in_data


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
    :return: the linear regression model
    """
    reg = linear_model.Ridge()
    reg.fit(x, y)
    return reg


def normalize(x: pd.DataFrame):
    """
    Scale all input features to the range [0, 1].
    :param x: input features
    :return: tuple: (a numpy numerical matrix with normalized features inside, price)
    """
    date = x['日期']
    pig_num = x['生猪存栏（万头）']
    sow_num = x['母猪存栏（万头）']
    price = x['苗猪平均价格（元/公斤）']

    # hint: use timestamp() method of pandas.Timestamp to convert date to int
    date = list(map(lambda t: t.timestamp(), date.tolist()))

    norm = lambda a: (a - np.min(a)) / (np.max(a) - np.min(a))

    date = norm(date).reshape(-1, 1)
    pig_num = norm(pig_num).reshape(-1, 1)
    sow_num = norm(sow_num).reshape(-1, 1)

    return np.hstack((date, pig_num, sow_num)), price


def predict(x: np.ndarray, model: linear_model.Ridge) -> np.ndarray:
    """
    Predict the price
    :param x: normalized input features
    :param model: trained linear model
    """
    # TODO YOUR CODE HERE
    return model.predict(x)


def test_model():
    # TODO merge the whole process!
    local_dir = os.path.dirname(__file__)
    data_path = os.path.join(local_dir, 'data')  # the data directory

    # read in data from files. load training data and testing data
    data = read_in_data(data_path)
    data_train, _, data_test = split_data(data)

    # normalize data
    data_train = normalize(data_train)
    data_test = normalize(data_test)

    # train a linear model
    model = train(data_train[0], data_train[1])
    predicted_price = predict(data_test[0], model)

    # draw 2 different line to compare
    plt.figure(1)
    line_true = plt.scatter(data_test[0][:, 0], data_test[1], label='true price')
    line_predict = plt.scatter(data_test[0][:, 0], predicted_price, label='predict price')
    plt.legend(handles=[line_true, line_predict])
    plt.show()
    pass


if __name__ == '__main__':
    test_model()