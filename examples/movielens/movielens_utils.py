import pandas as pd
import numpy as np
import torch as t
def get_features():
    feats = pd.read_csv('ml-100k/u.item', sep='|', index_col=0, header=None, encoding='latin-1')
    feats = feats.drop([1,2,3,4,5], axis=1)
    feats.columns = ['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    feats = feats.to_numpy().repeat(943)
    feats = feats.reshape(943,1682,18)
    return t.tensor(feats).type(t.FloatTensor)


def get_ratings():
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', header=None)
    ratings.columns = ['user id', 'item id',  'rating',  'timestamp']
    ratings['rating'].loc[ratings['rating'] >= 4] = 1
    ratings['rating'].loc[ratings['rating'] < 4] = 0
    ratings = ratings.pivot(index='user id', columns='item id', values='rating').fillna(0)
    return t.tensor(ratings.to_numpy())

# feats = get_features()
# print(feats.shape)
#
# ratings = get_ratings()
# print(ratings.shape)
