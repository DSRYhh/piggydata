import glob
import os
import pandas as pd


def read_in_data(path: str) -> pd.DataFrame:
    """
    Read in all data in path (in *.xls format)
    :param path: the path of data directory
    :return a pandas DataFrame contains all data and sorted by time
    """

    # list all file in path directory
    os.chdir(path)
    file_list = []
    for file in glob.glob("*.xls"):
        file_list.append(file)

    # Notice: the first line "上海市生猪养殖情况一览表" in each file should be abandoned
    data_frames = []
    for file in file_list:
        # parse date column as date type
        df = pd.io.excel.read_excel(file, sheetname=0, skiprows=1)
        data_frames.append(df)

    data_all = pd.DataFrame()
    data_all = pd.concat(data_frames)

    data_all.sort_values(['日期'], ascending=True, inplace=True)
    del data_all['序号']

    return data_all


if __name__ == '__main__':
    # TODO: Prerequisites
    # 1. Install Anaconda (https://repo.continuum.io/archive/Anaconda3-5.0.1-Windows-x86_64.exe)
    # 2. In PyCharm, open File - Settings - Project - Project Interpreter,
    #    in Project Interpreter, choose Anaconda3/python.exe

    # Notice backslash symbol in a string should be doubled
    # data_path = "C:\\Users\\19137\\Desktop\\pig\\shpig(1)\\shpig"
    data_path = "C:\\Users\\19137\\Desktop\\pig\\shpig(1)\\shpig\\data"
    data = read_in_data(data_path)
    print(data)
