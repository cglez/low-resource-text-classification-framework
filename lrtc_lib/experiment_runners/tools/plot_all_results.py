# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os

import seaborn as sns
import matplotlib.pyplot as plt

from lrtc_lib.active_learning.strategies import get_strategy_name_in_paper
from lrtc_lib.experiment_runners.experiment_runner import ExperimentRunner
from lrtc_lib.experiment_runners.experiments_results_handler import TRAIN_TOTAL_COUNT_HEADER
from lrtc_lib.experiment_runners.tools.merge_results import merge_results


sns.set_theme()
sns.set_palette("tab10")
palette = sns.color_palette()

dataset_names = {
    'ag_news': 'AG News',
    'ag_news_imbalanced_1': 'AG News (imb.)',
    'cola': 'CoLA',
    'imdb': 'IMDB',
    'isear': 'ISEAR',
    'polarity': 'Polarity',
    'polarity_imbalanced_positive': 'Polarity (imb.)',
    'subjectivity': 'Subjectivity',
    'subjectivity_imbalanced_subjective': 'Subjectivity (imb.)',
    'trec': 'TREC',
    'wiki_attack': 'Wiki Toxic',
}

colors = {
    'GREEDY_CORE_SET': palette[3],
    'DAL': palette[1],
    'DROPOUT_PERCEPTRON': palette[2],
    'HARD_MINING': palette[0],
    'PERCEPTRON_ENSEMBLE': palette[5],
    'RETROSPECTIVE': palette[6],
    'RANDOM': palette[4],
    'full': palette[7],
}

markers = {
    'GREEDY_CORE_SET': '^',
    'DAL': '*',
    'DROPOUT_PERCEPTRON': 's',
    'HARD_MINING': 'P',
    'PERCEPTRON_ENSEMBLE': 'D',
    'RETROSPECTIVE': 'p',
    'RANDOM': 'o',
    'full': '',
}

marker_sizes = {
    'GREEDY_CORE_SET': 6,
    'DAL': 8,
    'DROPOUT_PERCEPTRON': 6,
    'HARD_MINING': 8,
    'PERCEPTRON_ENSEMBLE': 6,
    'RETROSPECTIVE': 8,
    'RANDOM': 6,
    'full': 0,
}


def plot_results(write=False, full_line=True):
    #df_all = pd.read_csv('lrtc_lib/output/experiments/all_results.csv')
    df_all = merge_results()

    datasets = df_all["dataset"].unique()
    x_col = TRAIN_TOTAL_COUNT_HEADER

    for scenario in 'balanced', 'imbalanced', 'query':
        for dataset in datasets:
            df = df_all[(df_all["dataset"] == dataset) & (df_all['scenario'] == scenario)]
            metric = "accuracy" if scenario == "balanced" else "f1"
            models = sorted(df["model"].unique())

            for model in models:
                #if model == 'NB':
                #    continue
                model_df = df[df["model"] == model]
                if model_df.empty:
                    continue
                model_name = 'BERT' if model == 'HFBERT' else model

                full = df_all[(df_all['scenario'] == 'full') & (df_all["dataset"] == dataset)
                              & (df_all['model'] == model)]
                n_full = len(full)
                if full_line:
                    full = full.mean()
                    plt.axhline(full[metric], linestyle='dotted', color=colors['full'], label=None)

                print(scenario, dataset, model, len(model_df), f'+{n_full}')

                als = sorted(df["AL"].unique().tolist())
                als.remove('RANDOM')
                als += ['RANDOM']

                for al in als:
                    if al == ExperimentRunner.NO_AL:
                        continue
                    df_al = model_df[(model_df["AL"] == al) | (model_df["AL"] == ExperimentRunner.NO_AL)]
                    df_al = df_al[[x_col, metric]]
                    df_al = df_al.dropna(axis=0)
                    model_df_avg = df_al.groupby(x_col, as_index=False).mean()
                    if len(model_df_avg[x_col]) <= 1:
                        continue  # skip lines with a single point (e.g. incompatible AL-model pairs)
                    al_name = get_strategy_name_in_paper(al)
                    sns.lineplot(data=df_al, x=x_col, y=metric, label=al_name, color=colors[al], err_style=None,
                                 marker=markers[al], markersize=marker_sizes[al], markeredgewidth=0,
                                 linestyle='dashed' if al == "RANDOM" else 'solid')

                plt.title(dataset_names[dataset])
                plt.ylabel('F1' if metric == 'f1' else metric)
                plt.xlabel('# training samples')

                if write:
                    os.makedirs('lrtc_lib/output/figures', exist_ok=True)
                    plt.savefig(f"lrtc_lib/output/figures/{scenario}_{model_name.lower()}_{dataset}.pdf")
                else:
                    plt.show()
                plt.close()


def plot_legends(write=False):
    df_all = merge_results()

    models = sorted(df_all["model"].unique())
    for model in models:
        model_name = 'BERT' if model == 'HFBERT' else model

        als = df_all[df_all['model'] == model]['AL'].unique().tolist()
        als = [x for x in als if x not in ['no_active_learning', 'full_model']]
        als.remove('RANDOM')
        als += ['RANDOM']

        ax = plt.subplot()
        ax.set_axis_off()

        handles = []
        for al in als:
            line, = ax.plot([], color=colors[al], label=get_strategy_name_in_paper(al),
                            marker=markers[al], markersize=marker_sizes[al],
                            linestyle='dashed' if al == "RANDOM" else 'solid')
            handles.append(line)

        ax.legend(handles=handles)

        if write:
            os.makedirs('lrtc_lib/output/figures', exist_ok=True)
            plt.savefig(f"lrtc_lib/output/figures/legend_{model_name}.pdf")
        else:
            plt.show()
        plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Plot the results in the given CSV files, per dataset and metric")
    parser.add_argument('--write', default=False, action='store_true')
    args = parser.parse_args()

    plot_results(write=args.write)
