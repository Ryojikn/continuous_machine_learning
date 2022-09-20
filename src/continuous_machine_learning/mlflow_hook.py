from typing import Any, Dict
import mlflow
import mlflow.sklearn
from kedro.framework.hooks import hook_impl
from kedro.pipeline.node import Node
import logging
import os
from kedro.config import ConfigLoader
from kedro.framework.project import settings
from pathlib import Path

os.environ["DATABRICKS_HOST"] = "https://adb-5911813420488264.4.azuredatabricks.net/"
os.environ["MLFLOW_TRACKING_URI"] = "databricks"
if not os.getenv("DATABRICKS_TOKEN"):
    conf_path = Path(os.getcwd()) / settings.CONF_SOURCE
    conf_loader = ConfigLoader(conf_source=conf_path, env="local")
    credentials = conf_loader.get("credentials*", "credentials*/**")
    os.environ["DATABRICKS_TOKEN"] = credentials["databricks_token"]


class ModelTrackingHooks:
    """Namespace for grouping all model-tracking hooks with MLflow together."""

    def __init__(self):
        main_uri = "databricks"
        mlflow.set_tracking_uri(main_uri)
        mlflow.set_registry_uri(main_uri)
        mlflow.set_experiment("/Shared/cml")

    def __load_config__(self):
        conf_path = Path(os.getcwd()) / settings.CONF_SOURCE
        conf_loader = ConfigLoader(conf_source=conf_path, env="local")
        conf_params = conf_loader.get("parameters/train*")
        return conf_params

    @hook_impl
    def before_pipeline_run(self, run_params: Dict[str, Any]) -> None:
        """Hook implementation to start an MLflow run
        with the session_id of the Kedro pipeline run.
        """
        if os.getenv("ENVIRONMENT") in ["pre", "pro"]:
            with open("./.runid", "r") as filetoread:
                production_runid = filetoread.read()
                print(production_runid)
            run = mlflow.start_run(run_id=production_runid)
            print("Running with the following run_id: {run.info.run_id}")
        else:
            run = mlflow.start_run(run_name=run_params["session_id"])
            with open("./.runid", "w") as filetowrite:
                filetowrite.write(run.info.run_id)
            mlflow.log_params(run_params)

    @hook_impl
    def after_node_run(
        self, node: Node, outputs: Dict[str, Any], inputs: Dict[str, Any]
    ) -> None:
        """Hook implementation to add model tracking after some node runs.
        In this example, we will:
        * Log the parameters after the data splitting node runs.
        * Log the model after the model training node runs.
        * Log the model's metrics after the model evaluating node runs.
        """
        mlflow.autolog(log_models=False)
        if node._func_name == "split_data":
            conf_params = self.__load_config__()
            mlflow.log_params(
                {
                    "test_size": conf_params[node._namespace.split(".")[0]][
                        node._namespace.split(".")[1]
                    ]["model_options"]["test_size"]
                }
            )

        elif node._func_name == "train_model":
            model = outputs[f"{node._namespace}.regressor"]
            mlflow.sklearn.log_model(model, f"model_{node._namespace.split('.')[1]}")

        elif node._func_name == "evaluate_model":
            mlflow.log_metric(
                key=f"{node._namespace}_test_r2score",
                value=outputs[f"{node._namespace}.score"],
            )

    @hook_impl
    def after_pipeline_run(self) -> None:
        """Hook implementation to end the MLflow run
        after the Kedro pipeline finishes.
        """
        mlflow.end_run()
