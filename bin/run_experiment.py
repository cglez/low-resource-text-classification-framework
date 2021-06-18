# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import datetime
import logging
import os
import argparse
import json
from pathlib import Path
from collections import defaultdict

import lrtc_lib.experiment_runners.experiments_results_handler as res_handler
from lrtc_lib.experiment_runners.experiment_runner import ExperimentParams
from lrtc_lib.experiment_runners.experiment_runner_types import instantiate_experiment_runner
from lrtc_lib.experiment_runners.tools.plot_results import plot_results
from lrtc_lib.experiment_runners.core.save_config import save_config
from lrtc_lib.active_learning.strategies import ActiveLearningStrategies
from lrtc_lib.train_and_infer_service.model_type import ModelTypes
from lrtc_lib.orchestrator.orchestrator_api import get_workspace_id


if __name__ == '__main__':
    start_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="configuration JSON file")
    parser.add_argument("--name")
    parser.add_argument("--num_iterations", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--starting_repeat_id", type=int, default=1)
    parser.add_argument("--datasets")
    parser.add_argument("--models")
    parser.add_argument("--strategies")
    args = parser.parse_args()

    with open(args.config) as file:
        config = json.load(file)

    # define experiment parameters, with command arguments taking precedence
    if args.name is not None:
        experiment_name = args.name
    else:
        experiment_name = config['experiment_name']
    if args.num_iterations is not None:
        active_learning_iterations_num = args.num_iterations
    else:
        active_learning_iterations_num = config['active_learning_iterations_num']
    if args.repeats is not None:
        num_experiment_repeats = args.repeats
    else:
        num_experiment_repeats = config['num_experiment_repeats']
    # for full list of datasets and categories available run: python -m lrtc_lib.data_access.loaded_datasets_info
    datasets_categories_and_queries = config['datasets_categories_and_queries']
    if args.datasets is not None:
        datasets_categories_and_queries = {
            k: v for k, v in datasets_categories_and_queries.items() if k in args.datasets.split(',')
        }
    classification_models = [getattr(ModelTypes, model) for model in config['classification_models']]
    if args.models is not None:
        classification_models = [model for model in classification_models if model.name in args.models.split(',')]
    train_params = {model: config['classification_models'][model.name] for model in classification_models}
    active_learning_strategies = [getattr(ActiveLearningStrategies, al) for al in config['active_learning_strategies']]
    if args.strategies is not None:
        active_learning_strategies = [
            strategy for strategy in active_learning_strategies if strategy.name in args.strategies.split(',')
        ]

    experiment_runner = instantiate_experiment_runner(config)

    results_file_path, results_file_path_aggregated = res_handler.get_results_files_paths(
        experiment_name=experiment_name, start_timestamp=start_timestamp, repeats_num=num_experiment_repeats)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        filename=os.path.join(Path(results_file_path).parent, 'experiment.log'))

    save_config(
        str(Path(results_file_path).parent),
        experiment_name,
        experiment_runner,
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
                for repeat in range(args.starting_repeat_id, num_experiment_repeats + 1):
                    config = ExperimentParams(
                        experiment_name=experiment_name,
                        train_dataset_name=dataset + '_train',
                        dev_dataset_name=dataset + '_dev',
                        test_dataset_name=dataset + '_test',
                        category_name=category,
                        workspace_id=get_workspace_id(experiment_name, dataset, category, model.name, repeat),
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
