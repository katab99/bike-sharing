import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR


MODEL_REGISTRY = {
    "RandomForest": RandomForestRegressor,
    "LogisticRegression": LogisticRegression,
    "SVM": SVR,
}


def model_load(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["type"]
    model_params = config["model"]["params"]

    model_class = MODEL_REGISTRY.get(model_name)

    if not model_class:
        raise ValueError(f"Model '{model_name}' not found in registry.")

    model = model_class(**model_params)

    print(f"Initialized {model_name} with: {model_params}")
    return model
