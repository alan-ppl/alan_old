import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

data = pd.read_csv('Bus_Breakdown_and_Delays.csv', header=0, encoding='latin-1')
data = data.drop(['Schools_Serviced','Number_Of_Students_On_The_Bus','Has_Contractor_Notified_Schools','Has_Contractor_Notified_Parents','Have_You_Alerted_OPT','Informed_On','Incident_Number','Last_Updated_On', 'School_Age_or_PreK'], axis=1)
data['How_Long_Delayed'] = data['How_Long_Delayed'].map(lambda x: re.sub("[^0-9]", "", str(x)))
data = data.loc[data['How_Long_Delayed'] != '']
data['How_Long_Delayed'] = data['How_Long_Delayed'].map(lambda x: float(x))
data = data.loc[data['How_Long_Delayed'] < 130]
data = data.dropna().reset_index(drop=True)


ax = data['How_Long_Delayed'].astype(int).value_counts().sort_index().plot(kind='bar')
fig = ax.get_figure()
fig.savefig('figure.pdf')
