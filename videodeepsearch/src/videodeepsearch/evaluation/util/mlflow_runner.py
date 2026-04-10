import mlflow
from loguru import logger


def mlflow_setup(
    mlflow_tracking_url: str,
    experiment_name: str,
    experiment_tags: dict,
):
    mlflow.set_tracking_uri(mlflow_tracking_url)
    logger.info(f"MLflow tracking URI set to: {mlflow_tracking_url}")
    mlflow.set_experiment(experiment_name)
    mlflow.set_experiment_tags(experiment_tags)
    mlflow.agno.autolog() #type:ignore
    logger.info(f"Experiment set to: {experiment_name}")
    logger.info(f"Experiment tags set to: {experiment_tags}")
    logger.info("mlflow.log enabled")


