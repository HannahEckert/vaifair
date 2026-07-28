import subprocess
import sys
import os
import base64
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run recommendation loop and evaluation.')
    parser.add_argument('-n', type=int, default=30, help='Number of iterations to run')
    parser.add_argument('--dataset', '-d', type=str, default='example',
                        help='Name of the dataset (a subfolder under experiments/) to be evaluated')
    parser.add_argument('--model', '-m', type=str, default='BPR',
                        help='Name of RecBole model to be used')
    parser.add_argument('--choice-model', type=str, default='consume_all',
                        help='Name of choice model to be used.')
    parser.add_argument('-c', '--config', type=str, default='recbole_config_default.yaml',
                        help='Path to the Recbole config file')
    parser.add_argument('-a', '--artists-to-exclude', type=str, nargs='+', default=None,
                        help='List of artists to exclude from the recommendations. If None, no artists are excluded')
    parser.add_argument("--PP", action="store_true", help="Run post-processing after the recommendation loop")
    parser.add_argument("--PP-dimension", type=str, default="country", help="Dimension to consider for re-ranking (country or gender)")
    parser.add_argument("--PP-l", type=float, default=0.75, help="Trade-off parameter for post-processing")
    parser.add_argument("--PP-target-distribution", type=str, default="interactions", help="Target distribution for post-processing")
    parser.add_argument("--PP-seed", type=int, default=42, help="Random seed for shuffling user order in post-processing")

    return parser.parse_args()


def collect_results_as_dict(dataset):
    results_folder = os.path.join("experiments", dataset, "results")

    if not os.path.exists(results_folder):
        return {
            "dataset": dataset,
            "exists": False,
            "files": {}
        }

    files_content = {}
    for root, _, files in os.walk(results_folder):
        for file_name in sorted(files):
            file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(file_path, results_folder)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    files_content[relative_path] = {
                        "content_type": "text",
                        "content": f.read()
                    }
            except UnicodeDecodeError:
                with open(file_path, "rb") as f:
                    files_content[relative_path] = {
                        "content_type": "base64",
                        "content": base64.b64encode(f.read()).decode("ascii")
                    }

    # Reorganize files into nested structure
    nested_files = {
        "Embeddings": {},
        "Accepted_songs": {},
        "Proportions": {},
        "User": {},
        "Other": {}
    }
    
    for file_path, content in files_content.items():
        file_name = os.path.basename(file_path)
        
        if "embedding" in file_name:
            nested_files["Embeddings"][file_name] = content
        elif "accepted_songs" in file_name:
            nested_files["Accepted_songs"][file_name] = content
        elif "proportions" in file_name or "train_proportions" in file_name:
            nested_files["Proportions"][file_name] = content
        elif "user_statistics" in file_name:
            nested_files["User"][file_name] = content
        else:
            # Files that don't match any category go to Other
            nested_files["Other"][file_name] = content

    return {
        "dataset": dataset,
        "exists": True,
        "files": nested_files
    }

def call_script(n=30, dataset="example", model="BPR", choice_model="consume_all", config="recbole_config_default.yaml", 
                artists_to_exclude=None, PP=False, PP_dimension="country", PP_l=0.25,
                PP_target_distribution="interactions", PP_seed=42):
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
        if PP:
            command.append("--PP")
            command.extend(["--PP-dimension", PP_dimension])
            command.extend(["--PP-l", str(PP_l)])
            command.extend(["--PP-target-distribution", PP_target_distribution])
            command.extend(["--PP-seed", str(PP_seed)])
        result = subprocess.run(command, check=True)

        if result.returncode == 0:
            print(f"Iteration {i}: Success")
        else:
            print(f"Iteration {i}: Error")

    eval_command = [sys.executable, "evaluate.py", "--dataset", dataset]
    eval_result = subprocess.run(eval_command, check=True)

    if eval_result.returncode == 0:
        print("Evaluation: Success")
    else:
        print("Evaluation: Error")

    return collect_results_as_dict(dataset)


if __name__ == "__main__":
    args = parse_args()
    call_script(
        n=args.n,
        dataset=args.dataset,
        model=args.model,
        choice_model=args.choice_model,
        config=args.config,
        artists_to_exclude=args.artists_to_exclude,
        PP=args.PP,
        PP_dimension=args.PP_dimension,
        PP_l=args.PP_l,
        PP_target_distribution=args.PP_target_distribution,
        PP_seed=args.PP_seed,
    )
