#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# In[4]:


users = pd.read_csv("usersha1-profile.tsv", sep='\t')


# In[5]:


artist = pd.read_csv("usersha1-artmbid-artname-plays.tsv",
                     delimiter="\t",
                     header = None,
                     nrows = 2e6,
                    names = ['users', 'musicbrainz-artist-id', 'artist-name', 'plays'],
                          usecols = ['users','musicbrainz-artist-id', 'artist-name', 'plays'])


# In[4]:


# df.nonhash .apply(' '.join).str.replace('[^A-Za-z\s]+', '').str.split(expand=False)


# In[8]:


users[users['country'] =='United States']


# In[5]:


# artist.to_csv("userid_artist.csv")


# In[6]:


#To remove the non english worrds
# artist = artist.nonhashtag.apply(' '.join).str.replace('[^A-Za-z\s]+', '').str.split(expand=False)

# function to remove non-ASCII
# def remove_non_ascii(text):
#     return ''.join(i for i in text if ord(i)>48)

# artist['name'] = artist['name'].apply(remove_non_ascii)


# In[7]:


# function to remove non-ASCII
# def remove_non_ascii(text):
#     return ''.join(i for i in text if ord(i)>48)

# df['name'] = df['name'].apply(remove_non_ascii)


# In[8]:


users_artist = pd.read_csv("userid_artist.csv",names = ["users", "artistID", "artist_name", "plays"])


# In[9]:


# artist.iloc[-10:]
# users_artist.artist_name.isnull() == True
# users_artist.isnull().sum(axis=0)
if users_artist['artist_name'].isnull().sum() > 0:
    users_artist = users_artist.dropna(axis = 0, subset = ['artist_name'])


# In[10]:


# Read short data-------------------------------------------------START------------------------------


# In[11]:


# Read short data
users = users = pd.read_csv("new_user_data.csv",nrows=40970)
user_artist = pd.read_csv("userid_artist.csv")


# In[12]:


###### Drop unnamed column from user_artist
user_artist.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
user_artist.drop(["a"], axis=1, inplace=True)

users.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
users.drop(["a"], axis=1, inplace=True)


# In[13]:


users.rename({"userid":"userId"}, axis="columns", inplace=True)


# In[14]:


# Data Cleaning
# users.loc[users['age']>100].count() #130 people/40969
print("total values in age column",users['age'].count())
print(users['age'].loc[(users['age']>=10) & (users['age']<15)].count())
print(users['age'].loc[(users['age']>=15) & (users['age']<20)].count())
print(users['age'].loc[(users['age']>=20) & (users['age']<25)].count())
print(users['age'].loc[(users['age']>=25) & (users['age']<30)].count())

print(users['age'].loc[(users['age']>=30) & (users['age']<35)].count())
print(users['age'].loc[(users['age']>=35) & (users['age']<40)].count())
print(users['age'].loc[(users['age']>=40) & (users['age']<45)].count())
print(users['age'].loc[(users['age']>=45) & (users['age']<50)].count())
print(users['age'].loc[(users['age']>=50) & (users['age']<55)].count())
print(users['age'].loc[(users['age']>=55) & (users['age']<60)].count())
print(users['age'].loc[(users['age']>=60) & (users['age']<65)].count())

print("age is 25",users['age'].loc[(users['age'] == 25)].count())
print("age greater than 100",users['age'].loc[(users['age']>100)].count())


# In[15]:


users.age.describe()
# keep the age value between 0 - 100
users.loc[(users['age'] < 0) | (users['age'] > 100),['age']] = np.nan


# In[19]:


users['age'].isnull().sum(axis=0)
users['age'].dropna()
# Fill the Null values of age with mean value
users['age'] = users['age'].fillna(users['age'].mean())


# In[20]:


################### artist ID######################################
# user_artist.columns
user_artist['musicbrainz-artist-id'].isnull().sum(axis=0)
### remove null values for artist
user_artist = user_artist.loc[user_artist['musicbrainz-artist-id'].notnull()]


# In[27]:


user_artist.rename({"users":"userId"}, axis="columns", inplace=True)


# In[29]:


########################## Merge the two df #####################
user_artist = pd.merge(user_artist, users, on='userId', how='left')
# user_artist.to_csv("merged-file.csv")


# In[31]:


sns.factorplot(data=users,x='gender',kind='count')


# In[32]:


fig = plt.figure(figsize=(10,5))
users.country.value_counts().sort_values(ascending=False).head(15).plot(kind='bar')
plt.title("Users of top 15 Countries")
plt.ylabel("Number of Users")
plt.xlabel("Countries")


# In[33]:


#age
users = users.loc[(users['age']>=15) & (users['age']<100)]


# In[34]:


fig = plt.figure(figsize=(10,5))
users.age.value_counts().sort_values(ascending=False).head(15).plot(kind='bar')
plt.title("User counts based on age")
plt.ylabel("Number of Users")
plt.xlabel("Age")


# In[35]:


# top artists
fig = plt.figure(figsize=(10,5))
artist['artist-name'].value_counts().sort_values(ascending=False).head(15).plot(kind='bar')
plt.title("Top 15 Artists")
plt.ylabel("Number of Plays")
plt.xlabel("Artists Name")

