# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os

import pandas as pd

input_dir_path = './imdb'
parts = ['train', 'dev', 'test']

for part in parts:
    partition = pd.read_csv(os.path.join(input_dir_path, part + '.tsv'), sep='\t', names=['text', 'label'])
    print(f'{part} size = {len(partition)}')
    partition.to_csv(os.path.join(input_dir_path, part + '.csv'), columns=['label', 'text'])
