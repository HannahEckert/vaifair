import os.path
import argparse

from helper_files.data_loader import evaluate
experiments_to_evaluate = [
    'babyLFM5k',
    # add more experiments here
]


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate one or more experiments.')
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Dataset/experiment name (subfolder under experiments/) to evaluate.'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    target_experiments = [args.dataset] if args.dataset else experiments_to_evaluate

    for experiment in target_experiments:
        if not os.path.exists(os.path.join('experiments', experiment)):
            print(f"ERROR: Skipping {experiment}: Experiment does not exist!")
            continue
        print(f'Processing experiment "{experiment}"')
        experiment_folder = os.path.join('experiments', experiment)

        # delete previous results folder if it exists and create new one
        results_folder = os.path.join(experiment_folder, 'results')
        if os.path.exists(results_folder):
            print(f"Deleting previous results folder for {experiment}...")
            import shutil
            shutil.rmtree(results_folder)
        os.makedirs(results_folder, exist_ok=True)

        evaluate(experiments_folder='experiments', experiment_name=experiment)
