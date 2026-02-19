import mlflow
import mlflow.sklearn as mlflow_sklearn
import subprocess
from sklearn.metrics import root_mean_squared_error

from data.data_load import data_load
from model.model_load import model_load


mlflow.set_experiment("Bike_Sharing_Demand")

commit_hash = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
)


with mlflow.start_run():
    mlflow.set_tag("data_version", commit_hash)

    X_train, y_train = data_load(
        config_path="src/ml/data/data_config.yaml", type="train"
    )
    X_test, y_test = data_load(config_path="src/ml/data/data_config.yaml", type="test")

    model = model_load("src/ml/model/model_config.yaml")
    model.fit(X_train, y_train)

    # 3. ---
    rmse = root_mean_squared_error(y_test, model.predict(X_test))

    mlflow.log_metric("rmse", rmse)
    mlflow_sklearn.log_model(model, "bike_model")

    print(f"Run completed. RMSE: {rmse}")
