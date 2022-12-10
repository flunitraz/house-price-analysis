# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 13:37:37 2022

@author: Jarvi
"""

import pandas as pd
import numpy as np

df = pd.read_csv('../data/zingat_preprocessed.csv')
df.drop(['Unnamed: 0'], axis=1, inplace = True)

df

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

fiyat = df.iloc[:,9:10]

sol = df.iloc[:,:9]
sag = df.iloc[:,10:]

sol.drop(['isitma','esya','kullanim_durumu'], axis=1, inplace = True)
print(sol)

sag.drop(['il','ilce','mahalle','oda_sayisi'], axis=1, inplace = True)
print(sag)

veri = pd.concat([sol,sag],axis=1)
print(veri)

x_train, x_test, y_train, y_test = train_test_split(veri, fiyat, test_size = 0.33, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

print(y_test)
