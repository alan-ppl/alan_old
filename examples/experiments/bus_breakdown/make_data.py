import pandas as pd
import numpy as np
import torch as t
import re
from alan.experiment_utils import seed_torch


seed_torch(0)

def get_data():
    data = pd.read_csv('data/Bus_Breakdown_and_Delays.csv', header=0, encoding='latin-1')
    data = data.drop(['Schools_Serviced','Number_Of_Students_On_The_Bus','Has_Contractor_Notified_Schools','Has_Contractor_Notified_Parents','Have_You_Alerted_OPT','Informed_On','Incident_Number','Last_Updated_On', 'School_Age_or_PreK'], axis=1)
    data['How_Long_Delayed'] = data['How_Long_Delayed'].map(lambda x: re.sub("[^0-9]", "", str(x)))
    data = data.loc[data['How_Long_Delayed'] != '']
    data['How_Long_Delayed'] = data['How_Long_Delayed'].map(lambda x: float(x))
    data = data.loc[data['How_Long_Delayed'] < 130]
    data = data.dropna().reset_index(drop=True)
    return data



M = 3
J = 3
I = 60

df = get_data()
# df = data.pivot(index='School_Year', columns='Busbreakdown_ID', values='How_Long_Delayed').fillna(0)




# df_read = df.groupby(['School_Year', 'Boro', 'Busbreakdown_ID']).apply(lambda x: len(x['How_Long_Delayed'].unique())>N)
# zips = df_read[df_read].index.get_level_values('Busbreakdown_ID').to_list()
# df_read = df_read[df_read]
# print(df_read)
# print(zips)

df = df.reset_index()
df_id = df.groupby(['School_Year', 'Boro']).apply(lambda x: len(x['Busbreakdown_ID'].unique())>I)
boros = df_id[df_id].index.get_level_values('Boro').to_list()
df_id = df_id[df_id]
# print(df_id)
# print(boros)


df_id = df_id.reset_index()
df_boro = df_id.groupby('School_Year').apply(lambda x: len(x['Boro'].unique())>J)
years = df_boro[df_boro].index.get_level_values('School_Year').to_list()
df_boro = df_boro[df_boro]
# print(df_boro)
# print(years)



random_years = np.random.choice(years, M, replace=False)
df = df.loc[df['School_Year'].isin(random_years)]



new_boros = []
for m in df['School_Year'].unique():
    # print(m)
    # print(counties)
    # print(df.loc[df['stfips'] == m])
    # print(df.loc[df['stfips'] == m].loc[df['cntyfips'].isin(counties)]['cntyfips'].unique())
    new_boros.extend(np.random.choice(df.loc[df['School_Year'] == m].loc[df['Boro'].isin(boros) & ~df['Boro'].isin(new_boros)]['Boro'].unique(), J, replace=False))

df = df.loc[df['Boro'].isin(new_boros)]
num_boros = len(df['Boro'].unique())


new_ids = []
for j in df['Boro'].unique():

    new_ids.extend(np.random.choice(df.loc[df['Boro'] == j]['Busbreakdown_ID'].unique(), I, replace=False))

df = df.loc[df['Busbreakdown_ID'].isin(new_ids)]


num_ids = len(df['Busbreakdown_ID'].unique())

delay = df['How_Long_Delayed'].to_numpy()
df.drop(columns='How_Long_Delayed', inplace=True)
run_type = pd.get_dummies(df['Run_Type']).to_numpy()

Bus_Company_Name = pd.get_dummies(df['Bus_Company_Name']).to_numpy()
# print(df)


run_type = run_type.reshape(M, J, I, -1)

bus_company_name = Bus_Company_Name.reshape(M, J, I, -1)

delay = delay.reshape(M, J, I)
t.save(t.from_numpy(run_type)[:,:,:I//2,:], 'data/run_type_train.pt')
t.save(t.from_numpy(bus_company_name)[:,:,:I//2,:], 'data/bus_company_name_train.pt')
t.save(t.from_numpy(delay)[:,:,:I//2], 'data/delay_train.pt')

t.save(t.from_numpy(run_type)[:,:,I//2:,:], 'data/run_type_test.pt')
t.save(t.from_numpy(bus_company_name)[:,:,I//2:,:], 'data/bus_company_name_test.pt')
t.save(t.from_numpy(delay)[:,:,I//2:], 'data/delay_test.pt')
