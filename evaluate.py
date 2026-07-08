import os.path

from helper_files.data_loader import evaluate
experiments_to_evaluate = [
    'babyLFM5k',
    # add more experiments here
]

if __name__ == '__main__':
    for experiment in experiments_to_evaluate:
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
