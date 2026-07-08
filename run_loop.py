import subprocess
import sys

import argh
from argh import arg

@arg('-n', type=int, help='Number of iterations to run')
@arg('--dataset', type=str, help='Name of the dataset (a subfolder under experiments/) to be evaluated')
@arg('--model', type=str, help='Name of RecBole model to be used')
@arg('--choice-model', type=str, help='Name of choice model to be used.')
@arg('-c', '--config', type=str, help='Path to the Recbole config file')
@arg("--artists-to-exclude", type=str, nargs='+', default=None,
     help="List of artists to exclude from the recommendations. If None, no artists are excluded")

def call_script(n=30, dataset="example", model="BPR", choice_model="consume_all", config="recbole_config_default.yaml", artists_to_exclude=None):
    for i in range(1, n + 1):
        command = [
            sys.executable, "main.py", dataset, str(i),
            "--clean",
            "--model", model,
            "--choice-model", choice_model,
            "--config", config,
        ]
        if artists_to_exclude is not None:
            command.extend(["--artists-to-exclude"] + artists_to_exclude)
        result = subprocess.run(command, check=True)

        if result.returncode == 0:
            print(f"Iteration {i}: Success")
        else:
            print(f"Iteration {i}: Error")


if __name__ == "__main__":
    argh.dispatch_command(call_script)
