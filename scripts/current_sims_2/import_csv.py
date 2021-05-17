import csv

file_dir = "C:\\Users\\Christian\\Documents\\StudiumPhD\\Measurements\\"
file = file_dir + '2021-05-10_test_vac_0.04_mbar_fill.csv'
print(file)
with open(file
          , mode='r', encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file)
    data = list(map(list, zip(*csv_reader)))
    data_dict = {row[0]: row[1:] for row in data}
    print(data_dict)
    # mydict = {rows[0]:rows[1] for rows in csv_reader}
    # d = {}
    # for row in csv_reader:
    #     k, v = row
    #     d[k] = v


# # importing module 
# import csv
   
# # csv fileused id Geeks.csv
# filename="Geeks.csv"
  
# # opening the file using "with"
# # statement
# with open(filename,'r') as data:
#    for line in csv.reader(data):
#             print(line)
          
# # then data is read line by line 
# # using csv.reader the printed 
# # result will be in a list format 
# # which is easy to interpret