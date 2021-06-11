import sys
from lrtc_lib.experiment_runners.experiment_runners_core.plot_results import plot_results


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Invalid number of arguments.")
        print(f"Usage: {sys.argv[0]} <file.csv>")
        exit(1)
    plot_results(sys.argv[1])

