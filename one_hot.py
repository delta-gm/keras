# sklearn.preprocessing/one hot encoding/standardization

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import LabelEncoder # for one-hot encoding string values
import numpy as np 
import pandas as pd 

# classes(row11): ['compact', 'midsize', 'suv', '2seater', 'minivan', 'pickup', 'subcompact']

mpg_data = pd.read_csv('mpg.csv')
le=LabelEncoder()
mpg_data['class']=le.fit_transform(mpg_data['class'])
# print(mpg_data)

# one hot encoding for numerical values/string values must be label encoded first
column_transformer=ColumnTransformer([('encoder',OneHotEncoder(),[11])],remainder='passthrough')
onehot_mpg=np.array(column_transformer.fit_transform(mpg_data),dtype=np.str)
print(mpg_data[25:35])
print(onehot_mpg[25:35])
# mpg_columns = mpg_data.columns 




