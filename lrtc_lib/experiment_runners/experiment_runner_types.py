from lrtc_lib.experiment_runners.experiment_runner import ExperimentRunner
from lrtc_lib.experiment_runners.experiment_runner_balanced import ExperimentRunnerBalanced
from lrtc_lib.experiment_runners.experiment_runner_imbalanced import ExperimentRunnerImbalanced
from lrtc_lib.experiment_runners.experiment_runner_imbalanced_practical import ExperimentRunnerImbalancedPractical


experiment_runner_types = {
    "ExperimentRunnerBalanced": ExperimentRunnerBalanced,
    "ExperimentRunnerImbalanced": ExperimentRunnerImbalanced,
    "ExperimentRunnerImbalancedPractical": ExperimentRunnerImbalancedPractical,
}


def instantiate_experiment_runner(config: dict) -> ExperimentRunner:
    runner_type = config['runner']['type']
    runner_params = config['runner']['params']

    if runner_type == "ExperimentRunnerBalanced":
        experiment_runner = ExperimentRunnerBalanced(
                first_model_labeled_num=runner_params['first_model_positives_num'],
                active_learning_suggestions_num=runner_params['active_learning_suggestions_num'])
    elif runner_type == "ExperimentRunnerImbalanced":
        experiment_runner = ExperimentRunnerImbalanced(
                first_model_positives_num=runner_params['first_model_positives_num'],
                first_model_negatives_num=runner_params['first_model_negatives_num'],
                active_learning_suggestions_num=runner_params['active_learning_suggestions_num'])
    elif runner_type == "ExperimentRunnerImbalancedPractical":
        experiment_runner = ExperimentRunnerImbalancedPractical(
                first_model_labeled_from_query_num=runner_params['first_model_positives_num'],
                first_model_negatives_num=runner_params['first_model_negatives_num'],
                active_learning_suggestions_num=runner_params['active_learning_suggestions_num'],
                queries_per_dataset=config['datasets_categories_and_config'])
    else:
        raise(RuntimeError(f"Unknown experiment runner type '{runner_type}'."))

    return experiment_runner
