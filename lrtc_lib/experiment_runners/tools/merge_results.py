import glob
import os.path

import pandas as pd


def merge_results() -> pd.DataFrame:
    dfs = []

    for file in glob.glob('lrtc_lib/output/experiments/all_results/*.csv'):
        scenario = os.path.basename(file).split('_')[0]
        assert scenario in ['full', 'balanced', 'imbalanced', 'query', 'real'], f'Invalid scenario: {scenario}'

        df = pd.read_csv(file)
        df['scenario'] = scenario
        df['dataset'] = df['dataset'].str.replace('_test', '')
        df.rename(columns={'model id': 'model_id', 'category': 'label_class'}, inplace=True)
        dfs.append(df)

    combined = pd.concat(dfs, sort=True)
    cols = ['scenario', 'model', 'AL', 'dataset', 'label_class', 'iteration number', 'repeat id',
            'accuracy', 'f1', 'precision', 'recall', 'support', 'average_score',
            'tp', 'tn', 'fp', 'fn',
            'diversity', 'representativeness', 'query_matches',
            'train total count', 'train positive count', 'train negative count',
            'iteration runtime', 'selection runtime', 'train runtime', 'evaluation runtime',
            'seed', 'model_id',]
    combined = combined[cols]

    return combined


if __name__ == '__main__':
    combined = merge_results()
    print(combined)
    combined.to_csv('lrtc_lib/output/experiments/all_results.csv')
