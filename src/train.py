import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import subprocess
from load_data import load_data

mlflow.set_experiment("Bike_Sharing_Demand")

commit_hash = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
)


with mlflow.start_run():
    mlflow.set_tag("data_version", commit_hash)

    params = {"n_estimators": 100, "max_depth": 5}
    mlflow.log_params(params)

    X_train, X_test, y_train, y_test = load_data()
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    rmse = root_mean_squared_error(y_test, model.predict(X_test))
    mlflow.log_metric("rmse", rmse)

    mlflow.sklearn.log_model(model, "bike_model")

    print(f"Run completed. RMSE: {rmse}")
