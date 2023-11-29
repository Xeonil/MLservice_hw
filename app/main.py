from fastapi import FastAPI
from classes import Item
from classes import Items
from classes import MyModel
from classes import read_csv_and_convert
from typing import List
import pandas as pd


app = FastAPI()

model_instance = MyModel()

@app.get('/')
def root():
    return 'My first ML service'


@app.post("/predict_item")
def predict_item(item: Item) -> float:

    input_data = pd.DataFrame([item.dict()])
    prediction = model_instance.predict(input_data)

    return prediction[0]


@app.post("/predict_items")
def predict_items(variable):

    variable = read_csv_and_convert(variable) 

    input_data = pd.DataFrame(variable)
    prediction = model_instance.predict(input_data)

    return prediction