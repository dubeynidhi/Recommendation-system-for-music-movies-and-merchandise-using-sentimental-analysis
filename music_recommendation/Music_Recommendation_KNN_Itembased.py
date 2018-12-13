
import pandas as pd
import numpy as np
import sklearn
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import csv
import dask.dataframe as dd

# code for data cleaning

# Read files from dataset using pandas
user_data =pd.read_csv(r"usersha1-profile.tsv",delimiter='\t',encoding='utf-8')
user_data.columns = ['userid','gender','age','country', 'signup']

song_data = pd.read_csv(r"usersha1-artmbid-artname-plays.tsv",delimiter='\t',encoding='utf-8', header = None,nrows = 2e6,names = ['users', 'musicbrainz_artist_id', 'artist_name', 'plays'],usecols = ['users', 'musicbrainz_artist_id','artist_name', 'plays'])

# Display and confirm values for data
print(list(song_data.columns.values)) #file header
print(song_data)
print(song_data.head())
print(song_data.dtypes)
print(song_data.artist_name)

print(list(user_data.columns.values)) #file header
print(user_data.shape)
print(user_data)

#user data cleanup to remove sign up column as it was not required
user_data.drop(['signup'],axis=1,inplace=True)

# print entries where there is no age or gender specified

# print(user_data.age.isnull().sum())
# print(user_data.gender.isnull().sum())


# remove empty blocks in age by substtuting with mean
user_data.age=user_data.age.fillna(user_data.age.mean())
user_data.age=user_data.age.astype(np.int32)

# remove empty blocks in gender by substtuting with m
user_data.gender=user_data.gender.fillna('m')
user_data.gender=user_data.gender.astype(np.object)

# print again. Now the values should be 0
# print(user_data.age.isnull().sum())
# print(user_data.gender.isnull().sum())

# save file to another csv
user_data.to_csv(r"new_user_data.csv")


pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Read song data from dataset which has user id, artist id, artist, number of plays 
song_data = pd.read_csv('userid_artist.csv',
                          nrows = 2e7,sep=',',low_memory=False,header=None,dtype=str,
                           names = ['','users', 'musicbrainz-artist-id', 'artist-name', 'plays'],
                          usecols = ['users', 'musicbrainz-artist-id', 'artist-name', 'plays'])

# Replace invalid data
song_data = song_data.replace('!!!', 'Elvis Presley')
song_data = song_data.drop(song_data.index[0])
# song_data.columns = [song_data.replace('betty blowtorch', 'Elvis Presley') for x in song_data.columns]              
#print(song_data.head())
#print(song_data.dtypes)

# Read user data which has user id, age,gender, country
user_data = pd.read_csv('new_user_data.csv',
                          header = None,sep=',',low_memory=False,dtype=str,
                          names = ['','users', 'gender', 'age', 'country'],
                          usecols = ['users', 'gender', 'age','country'])

user_data = user_data.drop(user_data.index[0])
#print(user_data.head())

# calculate total number of plays for a artist and form it in a matrix artist_plays
artist_plays = (song_data.groupby(['artist-name'])['plays'].agg(lambda x: pd.to_numeric(x, errors='coerce').sum()).reset_index().rename(columns = {'plays': 'total_plays'})
     [['artist-name', 'total_plays']]
    )
# if(artist_plays.artist-name=='!!!'):
#     artist_plays.artist-name='Elvis Presley'
# artist_plays['artist-name'].replace(['!!!'], 'Elvis Presley')

# Merge it to combine the two arrays
song_data_with_artist_plays = song_data.merge(artist_plays)
song_data_with_artist_plays.head()

# see the distribution of the data
print(artist_plays['total_plays'].quantile(np.arange(.9, 1, .01))) 

# select artist who have more than 70000 plays
song_data_popular_artists = song_data_with_artist_plays.query('total_plays >= 70000')

# merge with user data and select on the basis of country. This is optional
sel_data = song_data_popular_artists.merge(user_data)
sel_data = sel_data.query('country == \'Canada\'')

# form a matrix with users as rows and artist as columns, number of plays will be the value
artist_data = sel_data.pivot(index = 'artist-name', columns = 'users', values = 'plays').fillna(0)
artist_data_sparse = csr_matrix(artist_data.values,dtype=int)

# Use KNN to model the data 
model_knn = NearestNeighbors(metric = 'cosine', algorithm='auto')
model_knn.fit(artist_data_sparse)

# now pick a artist randomly and suggest 5 artist similar to it 
ran_artist = np.random.choice(artist_data.shape[0])

distances, indices = model_knn.kneighbors(artist_data.iloc[ran_artist, :].values.reshape(1, -1), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for artist:{0}\n'.format(artist_data.index[ran_artist]).upper())
    else:
        print('{0}: {1}.'.format(i, artist_data.index[indices.flatten()[i]]).upper())

