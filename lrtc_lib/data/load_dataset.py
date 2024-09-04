# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import logging

from lrtc_lib.data_access import single_dataset_loader
from lrtc_lib.data_access.processors.dataset_part import DatasetPart
from lrtc_lib.oracle_data_access import gold_labels_loader


def load(dataset: str, force_new: bool = False):
    for part in DatasetPart:
        dataset_name = dataset + '_' + part.name.lower()
        # load dataset (generate Documents and TextElements)
        if force_new:
            single_dataset_loader.clear_all_saved_files(dataset_name)
        single_dataset_loader.load_dataset(dataset_name, force_new)
        # load gold labels
        if force_new:
            gold_labels_loader.clear_gold_labels_file(dataset_name)
        gold_labels_loader.load_gold_labels(dataset_name, force_new)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs='+', help="List of datasets to load")
    parser.add_argument("--force-new", action='store_true', help="Force a new loading of the datasets")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    for dataset_name in args.datasets:
        load(dataset=dataset_name, force_new=args.force_new)
