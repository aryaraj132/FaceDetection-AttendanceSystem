import csv
import os
import pandas as pd
from datetime import datetime

# now = datetime.now()
# date_time = now.strftime("%d/%m/%Y %H:%M:%S")
# date = now.strftime("%d/%m/%Y")
# dir = os.path.dirname(os.path.realpath(__file__))
# dataset_path = path = os.path.join(dir, 'dataset') 
# if not (os.path.isdir(dataset_path)):
#     os.mkdir(dataset_path)
# df = pd.read_csv(dir + '/students.csv')
# coll = ['0']*len(df['id'])
# print(df)
# df.loc[len(df.index)] = [8,'me']
# print(df)
# df['Attendance'] = coll
# df.to_csv(dir + '/students.csv', index=False)
# if date in df.columns:
#     if (int(df.loc[df['id'] == 7, date].iloc[0]))==0:
#         df.loc[df['id'] == 7, date]=1
#         df.to_csv(dir + '/students.csv', index=False)
#     else:
#         print("Already Exist")
# else:
#     df[date] = coll
#     df.loc[df['id'] == 7, date]=1
#     df.to_csv(dir + '/students.csv', index=False)
# exist = False
# for i in range(len(df['id'])):
#     if df['id'].iloc[i] == 8:
#         exist = True
# if not exist:
#     pass
    # with open(dir + '/students.csv', mode='a', newline='') as csv_file:
    #     fieldnames = ['id', 'name']
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #     writer.writerow({'id': "1", 'name': "Ankit Raj"})
    # csv_file.close()
