import mlflow
from mlflow import MlflowClient
from fastapi import FastAPI
import os
app = FastAPI()

TRACKING_URI = "http://localhost:5000"
model_name = "Bike-Demand"


# load model
client = MlflowClient()
model_metadata = client.get_latest_versions(model_name, stages=["None"])
latest_model = model_metadata[0]
print(latest_model)


# def predict(features):
#     preds = model.predict(features)
#     return preds
#
# @app.get("/")
# def read_root():
#     return {"Hello": "World"}
#
# @app.post("/predict")
# def predict_endpoint(features):
#     # prepare features
#
#     # predict
#
#     # format prediction
#
#     return