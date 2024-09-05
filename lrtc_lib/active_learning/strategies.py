# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from lrtc_lib.train_and_infer_service.model_type import ModelTypes


class ActiveLearningStrategy(object):

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        if isinstance(other, ActiveLearningStrategy):
            return self.name == other.name
        else:
            raise TypeError(f"comparing {other.__class__} to ActiveLearningStrategy is not allowed! ")

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)


class ActiveLearningStrategies(object):
    RANDOM = ActiveLearningStrategy("RANDOM")
    HARD_MINING = ActiveLearningStrategy("HARD_MINING")
    RETROSPECTIVE = ActiveLearningStrategy("RETROSPECTIVE")
    CORE_SET = ActiveLearningStrategy("CORE_SET")
    GREEDY_CORE_SET = ActiveLearningStrategy("GREEDY_CORE_SET")
    DAL = ActiveLearningStrategy("DAL")
    DROPOUT_PERCEPTRON = ActiveLearningStrategy("DROPOUT_PERCEPTRON")
    PERCEPTRON_ENSEMBLE = ActiveLearningStrategy("PERCEPTRON_ENSEMBLE")


embedding_based_strategies = {
    ActiveLearningStrategies.CORE_SET,
    ActiveLearningStrategies.DAL,
    ActiveLearningStrategies.GREEDY_CORE_SET,
    ActiveLearningStrategies.DROPOUT_PERCEPTRON,
    ActiveLearningStrategies.PERCEPTRON_ENSEMBLE,
}


def get_compatible_models(model_type, active_learning_strategy):
    embedding_based_models = {ModelTypes.HFBERT}
    all_models = ModelTypes.get_all_types()

    if active_learning_strategy in embedding_based_strategies:
        return model_type in embedding_based_models
    else:
        return model_type in all_models


strategy_names_in_paper = {
    ActiveLearningStrategies.RANDOM: "Random",
    ActiveLearningStrategies.HARD_MINING: "LC",
    ActiveLearningStrategies.RETROSPECTIVE: "EGL",
    ActiveLearningStrategies.CORE_SET: "",
    ActiveLearningStrategies.GREEDY_CORE_SET: "Core-Set",
    ActiveLearningStrategies.DAL: "DAL",
    ActiveLearningStrategies.DROPOUT_PERCEPTRON: "Dropout",
    ActiveLearningStrategies.PERCEPTRON_ENSEMBLE: "PE",
}


def get_strategy_name_in_paper(strategy_name: str) -> str:
    if hasattr(ActiveLearningStrategies, strategy_name):
        strategy = getattr(ActiveLearningStrategies, strategy_name)
        return strategy_names_in_paper[strategy]
    else:
        return strategy_name
