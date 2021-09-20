
from fastapi import FastAPI
from sklearn.linear_model import LinearRegression
import pandas as pd  
import pickle

model = pickle.load(open("LinearRegressionMultiple.sav", "rb"))

api = FastAPI()

@api.get("/")
async def racine():
    return {"message": "Bonjour!"}

@api.get("/predict/{curbweight}&{enginesize}")
async def predict(curbweight: int, enginesize: int):
    X = pd.DataFrame({'curbweight': [curbweight], "enginesize": [enginesize]})
    y_pred = model.predict(X)[0][0]
    return {"price": y_pred}
