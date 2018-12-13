#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
# get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
from sklearn.cluster import KMeans
import pickle


# In[4]:

print("Creating clusters for datapoints")
data = pd.read_csv("merged-file.csv")


# In[5]:


###### Drop unnamed column from data
data.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
data.drop(["a"], axis=1, inplace=True)


# In[6]:


# Remove null values 
data = data.loc[data['age'].notnull()]


# In[7]:


genderdict = {'m':0,'f':1}
 ### taken from internet for best accuracy and correctness
countrylist = ['United States', 'Germany', 'United Kingdom', 'Poland', 'Russian Federation', 'Brazil', 'Sweden', 'Spain', 'Finland', 'Netherlands', 'Italy', 'France', 'Canada', 'Australia', 'Turkey', 'Norway', 'Czech Republic', 'Ukraine', 'Japan', 'Belgium', 'Mexico', 'Argentina', 'Switzerland', 'Austria', 'Romania', 'Portugal', 'Bulgaria', 'Chile', 'Denmark', 'Colombia', 'Greece', 'Hungary', 'Latvia', 'Slovakia', 'Croatia', 'Serbia', 'Lithuania', 'Estonia', 'Ireland', 'New Zealand', 'Belarus', 'Israel', 'India', 'Venezuela', 'Indonesia', 'Singapore', 'Slovenia', 'Korea, Republic of', 'China', 'South Africa', 'Malaysia', 'Philippines', 'Peru', 'Thailand', 'Moldova', 'Costa Rica', 'Iceland', 'Taiwan', 'Paraguay', 'Bosnia and Herzegovina', 'Antarctica', 'Puerto Rico', 'Georgia', 'Macedonia', 'Uruguay', 'Honduras', 'Barbados', 'Kazakhstan', 'Andorra', 'Saudi Arabia', 'United States Minor Outlying Islands', 'Djibouti', 'Cocos (Keeling) Islands', 'Tunisia', 'Egypt', 'Bolivia', 'Panama', 'Brunei Darussalam', 'Iran, Islamic Republic of', 'Dominican Republic', 'El Salvador', 'Haiti', 'Ecuador', 'Guatemala', 'Morocco', 'Pakistan', 'Burkina Faso', 'Azerbaijan', 'Cambodia', 'Hong Kong', 'Viet Nam', 'United Arab Emirates', 'Jamaica', 'Faroe Islands', 'Somalia', 'Guinea-Bissau', 'Micronesia, Federated States of', 'Tuvalu', "Cote D'Ivoire", 'Libyan Arab Jamahiriya', 'Nicaragua', 'Kyrgyzstan', 'Malta', 'Bermuda', 'Luxembourg', 'Kuwait', 'Cyprus', 'Heard Island and Mcdonald Islands', 'Christmas Island', 'Cuba', 'Niue', 'Aruba', 'Vanuatu', 'Dominica', 'Holy See (Vatican City State)', 'Uzbekistan', 'Bhutan', 'Montenegro', 'Reunion', 'Fiji', 'Netherlands Antilles', 'Lebanon', 'Liechtenstein']
countrydict = dict(zip(countrylist, [i for i in range(len(countrylist))]))


def mapr1(key):
    return genderdict[key]

def mapr2(key):
    return countrydict[key]


# In[8]:


data['gender'] = data['gender'].map(genderdict)
data['country'] = data['country'].map(countrydict)


# In[9]:


#create a dataframe with three columns 'age', 'sex' and 'country'
data_new = data.filter(['gender','age','country'], axis=1)

# fill nan values of country column with mean value
data_new['country'] = data_new['country'].fillna(data_new['country'].mode()[0])


# In[8]:


#apply kmeans on the data using scikit learn
kmeans = KMeans(n_clusters=10, init='k-means++',max_iter=500).fit(data_new)


# In[9]:



import pickle
# now you can save it to a file
with open('kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)


# In[10]:


#load kmeans from system
# and later you can load it
with open('kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)


# In[11]:


# temp = np.array([[0,float(25),1]])
# temp = np.array(temp).reshape((len(temp), 1))
# temp = scaler.transform(temp)
# temp = temp.reshape(1,-1)


# In[12]:


# kmeans.cluster_centers_
# kmeans.labels_.shape


# In[11]:


data_predict = pd.read_csv("merged-file.csv")


# In[12]:


###### Drop unnamed column from data
data_predict.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
data_predict.drop(["a"], axis=1, inplace=True)


# In[13]:


# Remove null values 
data_predict = data_predict.loc[data_predict['age'].notnull()]


# In[14]:


data_predict['gender'] = data_predict['gender'].map(genderdict)
data_predict['country'] = data_predict['country'].map(countrydict)

# fill nan values of country column with mean value
data_predict['country'] = data_predict['country'].fillna(data_predict['country'].mode()[0])


# In[17]:


Type_new = pd.Series([])


# In[67]:


# def cluster_assignment(dataframe):
#     return kmeans.predict([data_predict['gender'], data_predict['age'], data_predict['country']])

# for index, row in data_predict.iterrows():
#    data_predict[index]['cluster'] = kmeans.predict([data_predict[index]['gender'], float(data_predict[index]['age']), data_predict[index]['country']])

for idx, row in data_predict[0:100000].iterrows():
#     print(data_predict.iloc[idx][5])
    print(idx)
    temp = np.array([[row['gender'],float(row['age']),row['country']]])
    temp = temp.reshape(1,-1)
    Type_new[idx]=kmeans.predict(temp)
data_predict.insert(7,'cluster_number',Type_new)

#     data_predict.insert(iloc=idx, column='A', value=row['age'])
#     print(row[index]['age'])
#     print(data_predict[index]['gender'], float(data_predict[index]['age']), data_predict[index]['country'])
# data_predict['cluster'] = data_predict.apply(cluster_assignment, axis=1)


# In[20]:


# data_predict['cluster_number']
# data_predict = data_predict.drop(columns=['cluster_number', 'cluster_number1'])


# In[ ]:


# data_predict['cluster_number'][:100000]
# data_predict.to_csv("cluster_number.csv")


# In[15]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


# In[19]:


#apply kmeans on the data using scikit learn
kmeans = KMeans(n_clusters=10).fit(data_new)


# In[16]:


#Load the cluster number data
data_predict_clusterno = pd.read_csv("cluster_number.csv")


# In[17]:


###### Drop unnamed column from data
data_predict_clusterno.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
data_predict_clusterno.drop(["a"], axis=1, inplace=True)


# In[18]:


g = data_predict_clusterno.groupby('cluster_number')
g_new = g[['musicbrainz-artist-id', 'plays', 'cluster_number']]


# In[19]:


final_df = []
i = 0
for cluster, cluster_df in g_new:
    cluster_df_modif = cluster_df.loc[:, ~cluster_df.columns.isin(['cluster_number', 'userId', 'artist-name', 'gender', 'age', 'country'])]
#     cluster_df_modif = cluster_df.loc[:, cluster_df.columns != 'cluster_number']
    f = cluster_df_modif.groupby('musicbrainz-artist-id').sum()
    f['cluster_number'] = cluster
#     print(f.head())
    if i == 0:
        final_df = f
    else:
        final_df = pd.concat([final_df, f])
    i += 1
        
print(final_df)  


# In[20]:


# data_predict['cluster_number'][:100000]
# final_df.to_csv("final_df.csv")


# In[21]:


# final_df.groupby('cluster_number').sort_values(by=['plays'])


# In[22]:


final_df = final_df.sort_values(['plays'],ascending=False).groupby('cluster_number').head(10)


# In[23]:


b = final_df.sort_values(['cluster_number'])


# In[24]:


result = pd.read_csv("sorted_df.csv")


# In[25]:


top10 = result[:10]


# In[44]:


# result_final = pd.read_csv("result.csv")


# In[26]:


print(data_predict[data_predict['musicbrainz-artist-id'] == 'f59c5520-5f46-4d2c-b2c4-822eabf53419'])


# In[ ]:




