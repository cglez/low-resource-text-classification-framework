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


def plot_metric(dataset, metric, df, output_path, plot_uncertainty=False, show=False):
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
            if al == ExperimentRunner.FULL_MODEL:
                full_model_df_all = df[(df["model"] == model) & (df["AL"] == ExperimentRunner.FULL_MODEL)]
                full_model_df = full_model_df_all[metric]
                top_line = full_model_df.mean()
                plt.axhline(top_line, linestyle='dashed')
                continue
            model_df_all = df[(df["model"] == model) & ((df["AL"] == al) | (df["AL"] == ExperimentRunner.NO_AL))]
            if len(model_df_all[model_df_all['AL'] != ExperimentRunner.NO_AL]) == 0:
                continue  # skip lines with just the initial model (e.g. incompatible AL-model pairs)
            model_df = model_df_all[[x_col, metric]]
            model_df = model_df.dropna(axis=0)
            al_name = get_strategy_name_in_paper(al)
            ax = sns.lineplot(data=model_df, x=x_col, y=metric, ci=95 if plot_uncertainty else None,
                              label=f"{model}:{al_name}", linestyle='solid' if al != 'RANDOM' else 'dashed')
    plt.title(dataset)
    plt.savefig(os.path.join(output_path, f"{dataset}_{metric}"))
    if show:
        plt.show()
    plt.close()


def plot_results(paths: list, metrics=None, output_dir=None, plot_uncertainty=False, show=False):
    default_metrics = ["accuracy", "precision", "recall", "f1"]
    if metrics is None:
        metrics = default_metrics
    if output_dir is None:
        output_dir = Path(paths[-1]).parent
    df = pd.concat((pd.read_csv(f) for f in paths))
    datasets = df["dataset"].unique()
    for dataset in datasets:
        sub_df = df[df["dataset"] == dataset]
        for metric in metrics:
            plot_metric(dataset, metric, sub_df, output_dir, plot_uncertainty=plot_uncertainty, show=show)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Plot the results in the given CSV files, per dataset and metric")
    parser.add_argument("csv_file", nargs="+")
    parser.add_argument("-o", "--output-dir", help="Output directory")
    parser.add_argument("--metrics", help="Comma-separated list of metrics to use")
    parser.add_argument("--with-uncertainty", action='store_true', help="Plot confidence intervals")
    parser.add_argument("--show", action='store_true', help="Show plots interactively")
    args = parser.parse_args()

    metrics = args.metrics.split(",") if args.metrics is not None else None
    plot_results(args.csv_file, metrics=metrics, output_dir=args.output_dir, plot_uncertainty=args.with_uncertainty,
                 show=args.show)
