import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Untitled.CSV")
df.head()

x=df[['weight_kg']]
y=df['height_cm']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
#standardization

from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test =scaler.transform(x_test)
# regression
from sklearn.linear_model import LinearRegression
regression=LinearRegression( )
regression.fit(x_train,y_train)
print(regression.coef_)
print(regression.intercept_)

plt.scatter(x_train,y_train)
plt.show()