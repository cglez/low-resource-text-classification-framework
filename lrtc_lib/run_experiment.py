# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import datetime
import logging
import random
import re
import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import lrtc_lib.experiment_runners.experiments_results_handler as res_handler
from lrtc_lib.experiment_runners.experiment_runner import ExperimentRunner, ExperimentParams
from lrtc_lib.experiment_runners.experiment_runner_types import instantiate_experiment_runner
from lrtc_lib.experiment_runners.experiment_runners_core.plot_results import plot_results
from lrtc_lib.experiment_runners.experiment_runners_core.save_config import save_config
from lrtc_lib.oracle_data_access import oracle_data_access_api
from lrtc_lib.active_learning.strategies import ActiveLearningStrategies
from lrtc_lib.data_access.core.data_structs import Label, TextElement
from lrtc_lib.orchestrator import orchestrator_api
from lrtc_lib.orchestrator.orchestrator_api import LABEL_NEGATIVE
from lrtc_lib.train_and_infer_service.model_type import ModelTypes


if __name__ == '__main__':
    start_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as file:
            config = json.load(file)
    else:
        print("Invalid number of arguments.")
        print(f"Usage: {sys.argv[0]} config.json")

    # define experiments parameters
    experiment_name = config['experiment_name']
    active_learning_iterations_num = config['active_learning_iterations_num']
    num_experiment_repeats = config['num_experiment_repeats']
    # for full list of datasets and categories available run: python -m lrtc_lib.data_access.loaded_datasets_info
    datasets_categories_and_queries = config['datasets_categories_and_queries']
    classification_models = [getattr(ModelTypes, model) for model in config['classification_models']]
    train_params = {
        getattr(ModelTypes, model): config['classification_models'][model] for model in config['classification_models']
    }
    active_learning_strategies = [
        getattr(ActiveLearningStrategies, strategy) for strategy in config['active_learning_strategies']
    ]

    experiment_runner = instantiate_experiment_runner(config)

    results_file_path, results_file_path_aggregated = res_handler.get_results_files_paths(
        experiment_name=experiment_name, start_timestamp=start_timestamp, repeats_num=num_experiment_repeats)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        filename=os.path.join(Path(results_file_path).parent, 'info.log'))

    save_config(Path(results_file_path).parent,
                experiment_name, experiment_runner,
                active_learning_iterations_num,
                num_experiment_repeats,
                datasets_categories_and_queries,
                classification_models,
                train_params,
                active_learning_strategies)

    for dataset in datasets_categories_and_queries:
        for category in datasets_categories_and_queries[dataset]:
            for model in classification_models:
                results_all_repeats = defaultdict(lambda: defaultdict(list))
                for repeat in range(1, num_experiment_repeats + 1):
                    config = ExperimentParams(
                        experiment_name=experiment_name,
                        train_dataset_name=dataset + '_train',
                        dev_dataset_name=dataset + '_dev',
                        test_dataset_name=dataset + '_test',
                        category_name=category,
                        workspace_id=f'{experiment_name}-{dataset}-{category}-{model.name}-{repeat}',
                        model=model,
                        active_learning_strategies=active_learning_strategies,
                        repeat_id=repeat,
                        train_params=train_params[model])

                    # key: active learning name, value: dict with key: iteration number, value: results dict
                    results_per_active_learning = experiment_runner.run(
                        config,
                        active_learning_iterations_num=active_learning_iterations_num,
                        results_file_path=results_file_path,
                        delete_workspaces=True)
                    for al in results_per_active_learning:
                        for iteration in results_per_active_learning[al]:
                            results_all_repeats[al][iteration].append(results_per_active_learning[al][iteration])

                # aggregate the results of a single active learning iteration over num_experiment_repeats
                if num_experiment_repeats > 1:
                    agg_res_dicts = res_handler.avg_res_dicts(results_all_repeats)
                    res_handler.save_results(results_file_path_aggregated, agg_res_dicts)
    plot_results(results_file_path)

