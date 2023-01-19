# Task

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
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7fd2bb29e810>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/pyunpack/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7fd2bb29e510>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/pyunpack/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7fd2bb277210>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/pyunpack/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7fd2bb277550>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/pyunpack/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7fd2bb277890>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/pyunpack/
ERROR: Could not find a version that satisfies the requirement pyunpack (from versions: none)
ERROR: No matching distribution found for pyunpack
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7f07495cdc50>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/patool/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7f07495cd250>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/patool/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7f07495a8e90>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/patool/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7f07495a8c90>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/patool/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7f07495a8cd0>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/patool/
ERROR: Could not find a version that satisfies the requirement patool (from versions: none)
ERROR: No matching distribution found for patool
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_17/3988217755.py in <module>
     10 get_ipython().system('pip install pyunpack')
     11 get_ipython().system('pip install patool')
---> 12 from pyunpack.cli import Archive
     13 os.system('apt-get install p7zip')
     14 print(os.getcwd())

ModuleNotFoundError: No module named 'pyunpack'
add Codeadd Markdown
import datatable as dt
directory = '/kaggle/working/'
Archive('/kaggle/input/kkbox-music-recommendation-challenge/train.csv.7z').extractall(directory)
Archive('/kaggle/input/kkbox-music-recommendation-challenge/test.csv.7z').extractall(directory)
Archive('/kaggle/input/kkbox-music-recommendation-challenge/songs.csv.7z').extractall(directory)
Archive('/kaggle/input/kkbox-music-recommendation-challenge/members.csv.7z').extractall(directory)
Archive('/kaggle/input/kkbox-music-recommendation-challenge/song_extra_info.csv.7z').extractall(directory)
​
train = dt.fread('./train.csv').to_pandas()
test = dt.fread('./test.csv').to_pandas()
songs = dt.fread('./songs.csv').to_pandas()
members = dt.fread('./members.csv').to_pandas()
​
print('Data loading completed!')
print(train.shape, test.shape, songs.shape, members.shape)
add Codeadd Markdown
print(train.columns)
print(test.columns)
print(songs.columns)
print(members.columns)
add Codeadd Markdown
song_cols = ['song_id', 'song_length', 'genre_ids', 'artist_name', 'composer', 'language']
train = train.merge(songs[song_cols], on='song_id', how='left')
test = test.merge(songs[song_cols], on='song_id', how='left')
​
mem_cols = ['msno', 'city', 'bd', 'gender']
train = train.merge(members[mem_cols], on='msno', how='left')
test = test.merge(members[mem_cols], on='msno', how='left')
​
for col in [['msno', 'song_id', 'source_system_tab', 'source_screen_name',
             'source_type', 'genre_ids', 'artist_name',
             'composer', 'language', 'city', 'gender']]:
            train[col] = train[col].astype('category')
            test[col] = test[col].astype('category')
add Codeadd Markdown
for col in train.columns:
    print(train[col].value_counts(), "\n")
​
    
train = train.drop(['bd', 'msno', 'song_length', 'source_system_tab'], axis = 1)
test = test.drop(['bd', 'msno', 'song_length', 'source_system_tab'], axis = 1)
add Codeadd Markdown
train.columns
add Codeadd Markdown
test.columns
add Codeadd Markdown
df_col = [ 'song_id', 'source_screen_name',
       'source_type', 'genre_ids', 'artist_name', 'language', 'city', 'gender']
train = train.drop(['composer'], axis=1)
test = test.drop(['composer'], axis=1)
from sklearn.preprocessing import LabelEncoder
​
for i in range(len(df_col)):
    train[df_col[i]] = LabelEncoder().fit_transform(train[df_col[i]])
    
for i in range(len(df_col)):
    test[df_col[i]] = LabelEncoder().fit_transform(test[df_col[i]])
add Codeadd Markdown
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
train = my_imputer.fit_transform(train)
​
my_imputer = SimpleImputer()
test = my_imputer.fit_transform(test)
​
​
add Codeadd Markdown
train
add Codeadd Markdown
test
add Codeadd Markdown
​
​
train = pd.DataFrame(train, columns = [ 'song_id', 'source_screen_name','source_type', 
                                       'target',  'genre_ids', 'artist_name', 'language', 
                                       'city', 'gender'])
test = pd.DataFrame(test, columns = ['id', 'song_id', 'source_screen_name','source_type', 
                                       'genre_ids', 'artist_name', 'language', 
                                       'city', 'gender'])
add Codeadd Markdown
test
add Codeadd Markdown
train = train.astype(int)
test = test.astype(int)
add Codeadd Markdown
from sklearn.ensemble import RandomForestClassifier
​
from sklearn.model_selection import train_test_split
​
X = train
X = X.drop(['target'], axis = 1)
y = train[['target']]
​
print(X.head())
print(y.head())
add Codeadd Markdown
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
clf = RandomForestClassifier(n_estimators = 16)
clf.fit(X_train, y_train.values.ravel())
y_pred = clf.predict(X_test)
from sklearn import metrics 
print()
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
add Codeadd Markdown
pred = clf.predict(test.drop(['id'], axis = 1))
​
​
subm = pd.DataFrame()
subm['id'] = test['id']
subm['target'] = pred
​
subm
add Codeadd Markdown
subm.to_csv('submission2.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
print('Done!')
add Codeadd Markdown
​
add Codeadd Markdown
