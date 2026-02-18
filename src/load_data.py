import pandas as pd


def load_data():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    X_train = train.drop(["dteday", "instant", "casual", "registered", "cnt"], axis=1)
    X_test = test.drop(["dteday", "instant", "casual", "registered", "cnt"], axis=1)

    y_train = train["cnt"]
    y_test = test["cnt"]

    return X_train, X_test, y_train, y_test
