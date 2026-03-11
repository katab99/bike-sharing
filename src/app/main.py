import mlflow
from mlflow import MlflowClient
from fastapi import FastAPI
import os
app = FastAPI()

model_name = "Bike-Demand"

# load model
client = MlflowClient()
model_metadata = client.get_latest_versions(model_name, stages=["None"])
latest_model = model_metadata[0].source
model = mlflow.sklearn.load_model(latest_model)
print(model)



# def predict(features):
#     preds = model.predict(features)
#     return preds

@app.get("/")
def read_root():
    return {"message": "Ciao."}
#
@app.post("/predict")
def post_predict(features):
    # prepare features

    # predict
    preds = model.fit(features)

    # format prediction
    print(preds)
    return preds