import pandas as pd
import numpy as np



M = 2
J = 2
I = 2
N = 4

df_srrs2 = pd.read_csv('srrs2.dat')
df_cty = pd.read_csv('cty.dat')



df_srrs2.rename(columns=str.strip, inplace=True)
df_cty.rename(columns=str.strip, inplace=True)

df_srrs2['state'] = df_srrs2['state2'].str.strip()
df_srrs2['county'] = df_srrs2['county'].str.strip()

# We will now join datasets on Federal Information Processing Standards
# (FIPS) id, ie, codes that link geographic units, counties and county
# equivalents. http://jeffgill.org/Teaching/rpqm_9.pdf
df_srrs2['fips'] = 1000 * df_srrs2.stfips + df_srrs2.cntyfips
df_cty['fips'] = 1000 * df_cty.stfips + df_cty.ctfips

df = df_srrs2.merge(df_cty[['fips', 'Uppm', 'lon', 'lat']], on='fips')
df = df.drop_duplicates(subset='idnum')


df['wave'].replace({'  .': '-1'}, inplace=True)
df['rep'].replace({' .': '-1'}, inplace=True)
df['zip'].replace({'     ': '-1'}, inplace=True)
# Compute log(radon + 0.1)
df["log_radon"] = np.log(df["activity"] + 0.0001)

# Compute log of Uranium
df["log_u"] = np.log(df["Uppm"]+0.0001)


# Let's map floor. 0 -> Basement and 1 -> Floor
df['basement'] = df['basement'].apply(lambda x: 1 if x == 'Y' else 0)


df = df[['stfips', 'cntyfips', 'fips', 'log_u', 'basement', 'log_radon', 'zip']]
df['cntyfips'] = df['stfips'].map(str) + df['cntyfips'].map(str)
df['zip'] = df['cntyfips'] + df['zip']

df['reading'] = df.groupby('zip').cumcount()+1
df['r'] = df['reading']
df['reading'] = df['zip'].map(str) + df['reading'].map(str)
# Sort values by floor

# Reset index
df = df.reset_index(drop=True)

df_read = df.groupby(['stfips', 'cntyfips', 'zip']).apply(lambda x: len(x['reading'].unique())>N)
zips = df_read[df_read].index.get_level_values('zip').to_list()
df_read = df_read[df_read]
# print(df_read)
# print(zips)

df_read = df_read.reset_index()
df_zip = df_read.groupby(['stfips', 'cntyfips']).apply(lambda x: len(x['zip'].unique())>I)
counties = df_zip[df_zip].index.get_level_values('cntyfips').to_list()
df_zip = df_zip[df_zip]
# print(df_zip)
# print(counties)


df_zip = df_zip.reset_index()
df_st = df_zip.groupby('stfips').apply(lambda x: len(x['cntyfips'].unique())>J)
states = df_st[df_st].index.get_level_values('stfips').to_list()
df_st = df_st[df_st]
# print(df_st)
# print(states)



random_states = np.random.choice(states, M, replace=False)
df = df.loc[df['stfips'].isin(random_states)]

print(df)

new_counties = []
for m in df['stfips'].unique():
    # print(m)
    # print(counties)
    # print(df.loc[df['stfips'] == m])
    # print(df.loc[df['stfips'] == m].loc[df['cntyfips'].isin(counties)]['cntyfips'].unique())
    new_counties.extend(np.random.choice(df.loc[df['stfips'] == m].loc[df['cntyfips'].isin(counties)]['cntyfips'].unique(), J, replace=False))

df = df.loc[df['cntyfips'].isin(new_counties)]
print(df)

new_zips = []
for j in df['cntyfips'].unique():
    # print(m)
    # print(counties)
    # print(df.loc[df['stfips'] == m])
    # print(df.loc[df['stfips'] == m].loc[df['cntyfips'].isin(counties)]['cntyfips'].unique())
    new_zips.extend(np.random.choice(df.loc[df['cntyfips'] == j].loc[df['zip'].isin(zips)]['zip'].unique(), I, replace=False))

df = df.loc[df['zip'].isin(new_zips)]
print(df)

new_readings = []
for n in df['zip'].unique():
    new_readings.extend(np.random.choice(df.loc[df['zip'] == n]['reading'].unique(), N, replace=False))

# readings = np.random.choice(df['reading'].unique(), N, replace=False)
df = df.loc[df['reading'].isin(new_readings)]
df['reading'] = df['r']
df.drop(columns='r')
print(df)
print(df.shape)
print(2*2*2*4)
