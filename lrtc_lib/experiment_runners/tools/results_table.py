import pandas as pd
from fire import Fire

from lrtc_lib.active_learning.strategies import get_strategy_name_in_paper
from lrtc_lib.experiment_runners.tools.merge_results import merge_results
from lrtc_lib.experiment_runners.tools.plot_all_results import dataset_names


datasets = [
    'trec', 'wiki_attack', 'isear', 'ag_news_imbalanced_1', 'polarity_imbalanced_positive',
    'subjectivity_imbalanced_subjective', 'ag_news', 'imdb', 'cola', 'polarity'
]


def alc():
    df_all = merge_results()
    #print(df_all['dataset'].unique())
    print(df_all.info())
    cols = [x for x in df_all.columns.to_list() if x not in ('repeat id', 'model_id', 'seed')]
    df_all = df_all[cols]
    df_all['AL'] = df_all['AL'].apply(get_strategy_name_in_paper)
    df_all['f1'] *= 100
    df_all['accuracy'] *= 100

    avg = df_all.groupby(['scenario', 'model', 'dataset', 'AL'], as_index=False).mean()
    avg = avg.round(2)
    std = df_all.groupby(['scenario', 'model', 'dataset', 'AL', 'iteration number']).std()
    std = std.groupby(['scenario', 'model', 'dataset', 'AL']).mean().reset_index()
    std = std.round(2)
    #print(avg.to_csv())

    #df2 = df_all[df_all['model'] == 'HFBERT']
    #df2['performance'] = df2['f1']
    #df2['performance'][df2['scenario'] == 'balanced'] = df2['accuracy'][df2['scenario'] == 'balanced']
    #df2 = df2.groupby(['scenario', 'AL'], as_index=False).mean()
    #print(df2[['scenario', 'AL', 'performance']])
    ##df2 = df2.groupby(['AL'], as_index=False).mean()
    ##print(df2[['AL', 'performance']])

    for model in df_all['model'].unique():
        for scenario in df_all['scenario'].unique():
            if scenario == 'full':
                continue

            metric = 'accuracy' if scenario == 'balanced' else 'f1'
            #metric = 'recall'
            df = None
            for dataset in datasets:  # df_all['dataset'].unique():
                df_avg = avg[(avg['model'] == model) & (avg['scenario'] == scenario) & (avg['dataset'] == dataset)]
                df_std = std[(std['model'] == model) & (std['scenario'] == scenario) & (std['dataset'] == dataset)]
                if df_avg.empty:
                    continue

                if df is None:
                    df = pd.DataFrame(index=df_avg['AL'])
                df0 = pd.DataFrame(index=df_avg['AL'])
                df0[dataset] = df_avg[['AL', metric]].set_index('AL')
                df0[f'{dataset}_std'] = df_std[['AL', metric]].set_index('AL')
                df0[dataset] = df0.apply(lambda x: f'{x[0]}\\std{{{x[1]}}}', axis=1)
                dataset_name = dataset_names.get(dataset, dataset)
                df[dataset_name] = df0[dataset]

            print(model, scenario, metric)
            #print(df.to_latex(escape=False))
            print(df)


if __name__ == '__main__':
    Fire(alc)
