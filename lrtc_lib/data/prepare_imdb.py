# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os

import pandas as pd

input_dir_path = './imdb'
output_dir_path = os.path.join('..', 'available_datasets', 'imdb')
parts = ['train', 'dev', 'test']

for part in parts:
    dataset_part = pd.read_csv(os.path.join(input_dir_path, part + '.tsv'), sep='\t', names=['text', 'label'])
    print(f'{part} size = {len(dataset_part)}')
    dataset_part.to_csv(os.path.join(output_dir_path, part + '.csv'), columns=['label', 'text'])
