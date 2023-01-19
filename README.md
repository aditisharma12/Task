import numpy as np
import pandas as pd
from sklearn import model_selection
import os
import sys
import shutil
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from lightfm import LightFM
import scipy.sparse as sp
!pip install pyunpack
!pip install patool
from pyunpack.cli import Archive
os.system('apt-get install p7zip')
print(os.getcwd())


import datatable as dt
directory = '/kaggle/working/'
Archive('/kaggle/input/kkbox-music-recommendation-challenge/train.csv.7z').extractall(directory)
Archive('/kaggle/input/kkbox-music-recommendation-challenge/test.csv.7z').extractall(directory)
Archive('/kaggle/input/kkbox-music-recommendation-challenge/songs.csv.7z').extractall(directory)
Archive('/kaggle/input/kkbox-music-recommendation-challenge/members.csv.7z').extractall(directory)
Archive('/kaggle/input/kkbox-music-recommendation-challenge/song_extra_info.csv.7z').extractall(directory)

train = dt.fread('./train.csv').to_pandas()
test = dt.fread('./test.csv').to_pandas()
songs = dt.fread('./songs.csv').to_pandas()
members = dt.fread('./members.csv').to_pandas()

print('Data loading completed!')
print(train.shape, test.shape, songs.shape, members.shape)

print(train.columns)
print(test.columns)
print(songs.columns)
print(members.columns)

song_cols = ['song_id', 'song_length', 'genre_ids', 'artist_name', 'composer', 'language']
train = train.merge(songs[song_cols], on='song_id', how='left')
test = test.merge(songs[song_cols], on='song_id', how='left')

mem_cols = ['msno', 'city', 'bd', 'gender']
train = train.merge(members[mem_cols], on='msno', how='left')
test = test.merge(members[mem_cols], on='msno', how='left')

for col in [['msno', 'song_id', 'source_system_tab', 'source_screen_name',
             'source_type', 'genre_ids', 'artist_name',
             'composer', 'language', 'city', 'gender']]:
            train[col] = train[col].astype('category')
            test[col] = test[col].astype('category')
            
            for col in train.columns:
    print(train[col].value_counts(), "\n")
    
    

    
train = train.drop(['bd', 'msno', 'song_length', 'source_system_tab'], axis = 1)
test = test.drop(['bd', 'msno', 'song_length', 'source_system_tab'], axis = 1)

train.columns
test.columns

df_col = [ 'song_id', 'source_screen_name',
       'source_type', 'genre_ids', 'artist_name', 'language', 'city', 'gender']
train = train.drop(['composer'], axis=1)
test = test.drop(['composer'], axis=1)
from sklearn.preprocessing import LabelEncoder

for i in range(len(df_col)):
    train[df_col[i]] = LabelEncoder().fit_transform(train[df_col[i]])
    
for i in range(len(df_col)):
    test[df_col[i]] = LabelEncoder().fit_transform(test[df_col[i]])
    
    
    from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
train = my_imputer.fit_transform(train)

my_imputer = SimpleImputer()
test = my_imputer.fit_transform(test)
train
test

train = pd.DataFrame(train, columns = [ 'song_id', 'source_screen_name','source_type', 
                                       'target',  'genre_ids', 'artist_name', 'language', 
                                       'city', 'gender'])
test = pd.DataFrame(test, columns = ['id', 'song_id', 'source_screen_name','source_type', 
                                       'genre_ids', 'artist_name', 'language', 
                                       'city', 'gender'])
                                       
                                       
 test
 
 train = train.astype(int)
test = test.astype(int)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

X = train
X = X.drop(['target'], axis = 1)
y = train[['target']]

print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
clf = RandomForestClassifier(n_estimators = 16)
clf.fit(X_train, y_train.values.ravel())
y_pred = clf.predict(X_test)
from sklearn import metrics 
print()
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

pred = clf.predict(test.drop(['id'], axis = 1))


subm = pd.DataFrame()
subm['id'] = test['id']
subm['target'] = pred

subm

subm.to_csv('submission2.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
print('Done!')
    
    
    
