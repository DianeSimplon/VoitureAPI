import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('c:\\Users\\Abize\\OneDrive\\Bureau\\Etienne_diane_phileas\\APICars\\data\\RAW\\cars.csv', index_col="car_ID")
df_copy = df.copy()

X = df[['curbweight','enginesize']]
y = df[['price']]

model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 3)
entrainement = model.fit(X_train, y_train)

model_multiple = LinearRegression().fit(X, y);

file = 'LinearRegressionMultiple.sav'
pickle.dump(model_multiple, open(file, 'wb'))