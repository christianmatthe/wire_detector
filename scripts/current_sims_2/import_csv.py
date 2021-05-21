import csv

file_dir = "C:\\Users\\Christian\\Documents\\StudiumPhD\\Measurements\\"
file = file_dir + '2021-05-10_test_vac_0.04_mbar_fill.csv'
print(file)
# with open(file
#           , mode='r', encoding="utf-8") as csv_file:
#     csv_reader = csv.reader(csv_file)
#     data = list(map(list, zip(*csv_reader)))
#     data_dict = {row[0]: row[1:] for row in data}
#     print(data_dict)

import pandas as pd

data_dict = pd.read_csv(file)
print(data_dict.keys())