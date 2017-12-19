import pandas as pd
import dataReader

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
    data=data.sample(frac=1)
    data_train=pd.DataFrame()
    data_validation=pd.DataFrame()
    data_test=pd.DataFrame()

    data_train=data[:int(len(data)*0.6)]
    data_validation=data[int(len(data)*0.6+1) : int(len(data)*0.8)]
    data_test=data[int(len(data)*0.8+1) :]

    return (data_train,data_validation,data_test)

data_path = "C:\\Users\\19137\\Desktop\\pig\\shpig(1)\\shpig\\data"
data = dataReader.read_in_data(data_path)
print(split_data(data))