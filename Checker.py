import traceback

import numpy as np
import os

from dataReader import read_in_data
from prediction import normalize, split_data, train, predict


def is_normalized(x: np.ndarray) -> bool:
    """
    Check if all elements in numpy array is in range [0, 1]
    :param x: numpy array to be checked
    """
    return (x <= 1).all() and (x >= 0).all()


def normalization_check():
    try:
        # construct data dir
        local_dir = os.path.dirname(__file__)
        data_path = os.path.join(local_dir, 'data')

        # read in data from files
        data = read_in_data(data_path)
        data_train, data_val, data_test = split_data(data)

        data_train, data_val, data_test = list(
            map(lambda data_set: normalize(data_set), (data_train, data_val, data_test)))

        assert is_normalized(data_train[0]), "data_train is not normalized"
        assert is_normalized(data_val[0]), "data_val is not normalized"
        assert is_normalized(data_test[0]), "data_test is not normalized"

        assert data_train[0].shape[1] == 3, "the column of data_train should be 3"
        assert data_train[0].shape[1] == 3, "the column of data_train should be 3"
        assert data_train[0].shape[1] == 3, "the column of data_train should be 3"
    except Exception:
        print(f'Test failed')
        print(f'Exception trace back:')
        print(traceback.print_exc())
    else:
        print('Normalization test passed.')


def predict_check():
    try:
        local_dir = os.path.dirname(__file__)
        data_path = os.path.join(local_dir, 'data')

        # read in data from files
        data = read_in_data(data_path)
        data_train, data_val, data_test = split_data(data)

        data_train, data_val, data_test = list(
            map(lambda data_set: normalize(data_set), (data_train, data_val, data_test)))

        model = train(data_train[0], data_train[1])
        coeff = model.coef_

        assert coeff.shape == (3, ), "The shape of coefficient matrix should be (3, )"

        predict(data_test, model)
    except Exception:
        print(f'Training test failed')
        print(f'Exception trace back:')
        print(traceback.print_exc())
    else:
        print('Training test passed.')
        print(f"Linear regression coefficient: ")
        print(coeff)
