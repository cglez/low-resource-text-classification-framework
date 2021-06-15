import argparse
from collections import defaultdict

from lrtc_lib.experiment_runners.experiments_results_handler import get_res_dicts, avg_res_dicts, save_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Averages several result csv files into one")
    parser.add_argument("in_csv", nargs='+')
    parser.add_argument("out_csv")
    args = parser.parse_args()

    res_dicts = defaultdict(lambda: defaultdict(list))
    for file_name in args.in_csv:
        res_dicts = get_res_dicts(file_name, append_to=res_dicts)
    avg_res_dicts = avg_res_dicts(res_dicts)
    save_results(args.out_csv, avg_res_dicts, mode='w')
