import pandas as pd
import glob, os


data_path = "C:\\Users\\19137\\Desktop\\shpig\\data"
os.chdir(data_path)
file_list = []
for file in glob.glob("*.xls"):
    file_list.append(file)
data_frames = []
for file in file_list:
    # YOUR CODE HERE
    x = pd.ExcelFile(file)
    data_frames.append(x.parse("Sheet1"))

data_frames[0].head()
