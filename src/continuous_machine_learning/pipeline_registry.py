"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from continuous_machine_learning.pipelines import data_processing as dp
from continuous_machine_learning.pipelines import train as tr


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    data_processing_pipeline = dp.create_pipeline()
    train_pipeline = tr.create_pipeline()

    return {
        "__default__": data_processing_pipeline + train_pipeline,
        "dp": data_processing_pipeline,
        "train": data_processing_pipeline + train_pipeline,
        "score": data_processing_pipeline + train_pipeline,
    }
