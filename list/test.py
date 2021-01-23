import csv
import os
import pandas as pd
dir = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(dir + '/students.csv')
exist = False
for i in range(len(df['id'])):
    if df['id'].iloc[i] == 8:
        exist = True
if not exist:
    with open(dir + '/students.csv', mode='a', newline='') as csv_file:
        fieldnames = ['id', 'name']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({'id': "1", 'name': "Ankit Raj"})
    csv_file.close()
