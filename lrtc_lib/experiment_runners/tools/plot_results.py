# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from lrtc_lib.active_learning.strategies import get_strategy_name_in_paper
from lrtc_lib.experiment_runners.experiment_runner import ExperimentRunner
from lrtc_lib.experiment_runners.experiments_results_handler import TRAIN_TOTAL_COUNT_HEADER

sns.set_theme()
sns.set_palette("tab10")


def plot_metric(dataset, metric, df, output_path):
    """
    Plot the results in the most basic fashion. No complex color markers labels etc.
    """
    models = sorted(df["model"].unique())
    als = sorted(df["AL"].unique())
    x_col = TRAIN_TOTAL_COUNT_HEADER
    for model in models:
        for al in als:
            if al == ExperimentRunner.NO_AL:
                continue
            model_df_all = df[(df["model"] == model) & ((df["AL"] == al) | (df["AL"] == ExperimentRunner.NO_AL))]
            model_df = model_df_all[[x_col, metric]]
            model_df = model_df.dropna(axis=0)
            model_df = model_df.groupby(x_col, as_index=False).mean()
            if len(model_df[x_col]) <= 1:
                continue  # skip lines with a single point (e.g. incompatible AL-model pairs)
            x = model_df[x_col]
            y = model_df[metric]
            al_name = get_strategy_name_in_paper(al)
            ax = sns.lineplot(x=x, y=y, label=f'{model}:{al_name}', linestyle='dashed' if al == "RANDOM" else 'solid')
    plt.title(dataset)
    plt.savefig(os.path.join(output_path, f"{dataset}_{metric}"))
    # plt.show()
    plt.close()


def plot_results(path, metrics=None):
    all_metrics = ["accuracy", "f1"]
    if metrics is None:
        metrics = all_metrics
    df = pd.read_csv(path)
    datasets = df["dataset"].unique()
    for dataset in datasets:
        sub_df = df[df["dataset"] == dataset]
        for metric in metrics:
            plot_metric(dataset, metric, sub_df, Path(path).parent)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Plot the results in a CSV file per dataset and metric")
    parser.add_argument("csv_file")
    args = parser.parse_args()

    plot_results(args.csv_file)
