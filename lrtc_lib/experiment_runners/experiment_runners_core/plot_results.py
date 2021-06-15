# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from lrtc_lib.active_learning.strategies import strategy_names_in_paper
from lrtc_lib.experiment_runners.experiment_runner import ExperimentRunner

sns.set_theme()
sns.set_palette("tab10")


def plot_metric(dataset, metric, df, output_path):
    """
    Plot the results in the most basic fashion. No complex color markers labels etc.
    """
    models = sorted(df["model"].unique())
    als = sorted(df["AL"].unique())
    x_col = "train total count"
    for model in models:
        for al in als:
            if al == ExperimentRunner.NO_AL:
                continue
            model_df_all = df[(df["model"] == model) & ((df["AL"] == al) | (df["AL"] == "no_active_learning"))]
            model_df = model_df_all[[x_col, metric]]
            model_df = model_df.dropna(axis=0)
            model_df = model_df.groupby(x_col, as_index=False).mean()
            x = model_df[x_col]
            y = model_df[metric]
            al_name = strategy_names_in_paper[al]
            ax = sns.lineplot(x=x, y=y, label=f'{al_name}_{model}')
    plt.title(dataset)
    plt.savefig(os.path.join(output_path, f"{dataset}_{metric}"))
    plt.show()
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
    import sys
    if len(sys.argv) != 2:
        print("Invalid number of arguments.")
        print(f"Usage: {sys.argv[0]} <file.csv>")
        exit(1)
    plot_results(sys.argv[1])
