import os
import json
from lrtc_lib.experiment_runners.experiment_runner import ExperimentRunner


def save_config(output_path: str, experiment_name: str, runner: ExperimentRunner,
                active_learning_iterations_num: int, num_experiment_repeats: int,
                datasets_categories_and_config: dict, classification_models: list,
                train_params: dict, active_learning_strategies: list):
    """
    Save experiment runner configuration as JSON.
    """
    config = {
        'experiment_name': experiment_name,
        'runner': {
            'type': type(runner).__name__,
            'params': {
                'first_model_positives_num': runner.first_model_positives_num,
                'first_model_negatives_num': runner.first_model_negatives_num,
                'active_learning_suggestions_num': runner.active_learning_suggestions_num,
            }
        },
        'num_experiment_repeats': num_experiment_repeats,
        'active_learning_iterations_num': active_learning_iterations_num,
        'active_learning_strategies': [
            al.name for al in active_learning_strategies
        ],
        'classification_models': {
            model.name: train_params[model] for model in train_params if model in classification_models
        },
        'datasets_categories_and_config': datasets_categories_and_config,
    }

    with open(os.path.join(output_path, 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)
