"""
This is a boilerplate pipeline 'score'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import score_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=score_data,
                inputs=["model_input_table", "feature_list", "model"],
                outputs=["scored_data"],
                name="score_data_node",
            )
        ]
    )
