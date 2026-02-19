import pandas as pd


def load_data(data_path: str, input_features: list[str], target_feature: str):
    data = pd.read_csv(data_path)

    X = data[input_features]
    y = data[target_feature]

    return X, y
