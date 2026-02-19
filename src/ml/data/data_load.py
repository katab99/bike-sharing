import yaml
import pandas as pd
from typing import Literal


def data_load(config_path: str, type: Literal["train", "test"]):
    if type not in ("train", "test"):
        raise ValueError(f"Invalid type: {type}. Expected 'train' or 'test.")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    input_features = config["data"]["input_features"]
    target_feature = config["data"]["target_feature"]
    data_path = config["data"][type]

    data = pd.read_csv(data_path)

    X = data[input_features]
    y = data[target_feature]

    return X, y
