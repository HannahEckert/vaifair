import argparse
import json
import logging
import shutil
from pathlib import Path

import argh
import numpy as np
import pandas as pd
from argh import arg
from recbole.config import Config
from recbole.quick_start import load_data_and_model
from tqdm import tqdm

from choice_models import accept_new_recommendations
from recbole_wrapper import run_recbole_experiment, get_recbole_scores, get_recbole_ndcg_per_user

import torch

EXPERIMENTS_FOLDER = Path('experiments')


def prepare_run(dataset_name: str, iteration: int, clean=False, artists_to_exclude=None) -> tuple[Path, Path, Path]:
    """
    Prepares dataset folder structure and asserts all required files are present.
    :param dataset_name: name of the dataset to be used
    :param iteration: iteration number
    :param clean: if True, deletes all files in the data/ output/ and logs/ folders that may be present
                  Only has an effect if iteration number is 1
    :param artists_to_exclude: list of artists to exclude from the recommendations. If None, no artists are excluded
    :returns: a tuple of paths containing:
      - interactions file for the correct iteration
      - demographics file
      - tracks file
    """
    # Python/Pathlib magic: the division operator on Path objects functions as os.path.join()
    experiment_folder = EXPERIMENTS_FOLDER / dataset_name
    if not experiment_folder.exists():
        raise FileNotFoundError(f'Dataset {dataset_name} not found')

    # Assert that the required files are present in input/
    if not (experiment_folder / 'input').exists():
        raise FileNotFoundError(f'Dataset invalid: Input folder missing')

    demographics_file = experiment_folder / 'input' / 'demographics.tsv'
    if not demographics_file.exists():
        raise FileNotFoundError(f'Dataset invalid: input/demographics.tsv file missing')

    tracks_file = experiment_folder / 'input' / 'tracks.tsv'
    if not tracks_file.exists():
        raise FileNotFoundError(f'Dataset invalid: input/tracks.tsv file missing')

    shutil.rmtree('recbole_tmp', ignore_errors=True)
    shutil.rmtree('saved', ignore_errors=True)
    # Create recbole_workdir and saved if it doesn't exist
    (Path('recbole_tmp') / 'dataset').mkdir(exist_ok=True, parents=True)
    (Path('saved')).mkdir(exist_ok=True, parents=True)

    if iteration == 1:
        input_inter_file = experiment_folder / 'input' / f'dataset.inter'
        if not input_inter_file.exists():
            raise FileNotFoundError(f'Dataset invalid: input/dataset.inter')
        
        if artists_to_exclude is not None:
            tracks = pd.read_csv(tracks_file, sep="\t", header=None, names=["track_id","artist","title","country","gender"])
            dataset = pd.read_csv(input_inter_file, sep="\t", header=0)
            ids_to_exclude = tracks[tracks['artist'].isin(artists_to_exclude)]['track_id'].tolist()
            dataset_filtered = dataset[~dataset['item_id:token'].isin(ids_to_exclude)]

            users_to_exclude = dataset_filtered["user_id:token"].value_counts()[dataset_filtered["user_id:token"].value_counts() < 3].index
            dataset_filtered = dataset_filtered[~dataset_filtered['user_id:token'].isin(users_to_exclude)]
            
            #save the filtered dataset to a new file and redirect input_inter_file to that file
            filtered_inter_file = experiment_folder / 'input' / f'dataset_filtered.inter'
            dataset_filtered.to_csv(filtered_inter_file, sep="\t", header=True, index=False)
            input_inter_file = filtered_inter_file

        if clean:
            shutil.rmtree(experiment_folder / 'datasets', ignore_errors=True)
            shutil.rmtree(experiment_folder / 'output', ignore_errors=True)
            shutil.rmtree(experiment_folder / 'log', ignore_errors=True)
        (experiment_folder / 'datasets').mkdir(exist_ok=True)
        (experiment_folder / 'output').mkdir(exist_ok=True)
        (experiment_folder / 'log').mkdir(exist_ok=True)

        # Copy the interactions file to datasets/iteration_1.inter
        shutil.copy(input_inter_file, experiment_folder / 'datasets' / f'iteration_{iteration}.inter')

    inter_file = experiment_folder / f'datasets' / f'iteration_{iteration}.inter'

    # Copy inter file into recbole_workdir/iteration_{iteration}/dataset.inter
    shutil.copy(inter_file, Path('recbole_tmp') / f'dataset' / 'dataset.inter')

    return inter_file, demographics_file, tracks_file


def cleanup(dataset_name: str, iteration: int):
    """Cleanup after iteration is complete"""
    # create subfolders for logs
    log_folder = EXPERIMENTS_FOLDER / dataset_name / 'log' / f'iteration_{iteration}'
    shutil.rmtree(log_folder, ignore_errors=True)
    log_folder.mkdir(exist_ok=True)
    logging.shutdown()
    # move any logs that were written into this folder
    for log_file in Path('log').iterdir():
        shutil.move(str(log_file), str(log_folder / log_file.name))

    # move anything in log_tensorboard to the correct folder
    for log_file in Path('log_tensorboard').iterdir():
        shutil.move(str(log_file), str(log_folder / log_file.name))

    # remove the saved, log, log_tensorboard and recbole_workdir folder
    #shutil.rmtree('saved', ignore_errors=True) # We want to keep the saved model for later analysis, so we don't delete it here
    shutil.rmtree('log', ignore_errors=True)
    shutil.rmtree('log_tensorboard', ignore_errors=True)
    shutil.rmtree('recbole_tmp', ignore_errors=True)


def compute_top_k_scores(scores, k=10, orig_user_ids=None):
    """
    Computes the top k scores per user, saves it into the output folder
    and returns a dataframe with the new interactions
    """
    n = len(scores)
    item_ids, score = [], []

    for s in tqdm(scores, desc=f'calculating top-{k} item_ids', smoothing=0):
        items = np.argsort(-s)[:k]

        for item in items:
            item_ids.append(item)
            score.append(s[item])

    if orig_user_ids is None:
        orig_user_ids = list(range(n))
    user_ids = []
    for uid in orig_user_ids:
        user_ids.extend([uid] * k)

    df = pd.DataFrame.from_dict({
        'user_id': user_ids,
        'item_id': item_ids,
        'rank': list(range(1, k + 1)) * n,
        'score': score
    })
    return df


@arg('dataset_name', type=str, help='Name of the dataset (a subfolder under experiments/) to be evaluated')
@arg('iteration', type=int,
     help='Iteration number. Recbole training will use data from dataset_<iteration-1> and output to dataset_<iteration>')
@arg('-m', '--model', type=str, help='Name of RecBole model to be used')
@arg('-cm', '--choice-model', type=str, help='Name of choice model to be used. See README for available options')
@arg('-c', '--config', type=str, help='Path to the Recbole config file')
@arg('-k', type=int, help='Number of items to be recommended per user')
@arg("--artists-to-exclude", type=str, nargs='+', default=None,
     help="List of artists to exclude from the recommendations. If None, no artists are excluded")
@arg('--clean', action=argparse.BooleanOptionalAction,
     help='If True, deletes all files in the data/ output/ and logs/ folders that may be present')


def do_single_loop(
        dataset_name, iteration, model='BPR', choice_model='random',
        config='recbole_config_default.yaml',
        k=10, artists_to_exclude=None,
        clean=False):
    """
    Executes a single iteration loop consisting of training, evaluation and the
    addition of new interactions by a choice model. This file only does a single loop at a time and needs to be called as a subprocess.
    This is due to weirdness (coming from Recbole?) leading to a memory leak and swiftly decreasing performance after a few iterations.
    """
    print(f'Preparing iteration {iteration} for dataset/experiment {dataset_name}...')
    data_path, demographics_path, tracks_path = prepare_run(dataset_name, iteration, clean, artists_to_exclude)
    dataset_df = pd.read_csv(data_path, sep='\t', header=1, names=['user_id:token', 'item_id:token'])
    demographics_df = pd.read_csv(demographics_path, sep='\t', header=None,
                                  names=['country', 'age', 'gender', 'registration_time'])
    tracks_df = pd.read_csv(tracks_path, sep='\t', header=None, names=['title', 'artist', 'country'])
    print(' Done!')


    if iteration == 1:
        # Write a small json file containing the parameters with which this command was called.
        # this can be accessed by our plotting functions later
        call_params = {
            'dataset_name': dataset_name,
            'iteration': iteration,
            'model': model,
            'choice_model': choice_model,
            'config': config,
        }
        with open(EXPERIMENTS_FOLDER / dataset_name / 'params.json', 'w') as f:
            json.dump(call_params, f, indent=2)

    config = Config(model=model, dataset='dataset', config_file_list=[config])
    
    # Load checkpoint from previous iteration if not the first iteration
    checkpoint_path = None
    if iteration > 1:
        checkpoint_path = str(EXPERIMENTS_FOLDER / dataset_name / 'output' / f'iteration_{iteration - 1}_checkpoint.pth')
        if not Path(checkpoint_path).exists():
            print(f'Warning: Previous checkpoint not found at {checkpoint_path}')
            checkpoint_path = None
        else:
            print(f'Loading checkpoint from previous iteration: {checkpoint_path}')

    run_recbole_experiment(model=model, dataset=dataset_name, config=config, checkpoint_path=checkpoint_path)


    # Attempt to make sure the model is garbage collected and doesn't leak memory
    del config

    # There should only be one model file in saved folder, get its path
    model_path = str(next(Path('saved').iterdir()))
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=model_path)

    # Obtain recommendation scores
    scores = get_recbole_scores(model, dataset, test_data, config)

    # Get original user token IDs in the same order as the rows in the scores matrix
    orig_user_ids = [int(k) for k in dataset.field2token_id['user_id'].keys() if k != '[PAD]']

    # Obtain top k scores and save them for later analysis
    top_k_df = compute_top_k_scores(scores, k=k, orig_user_ids=orig_user_ids)
    top_k_df.to_csv(
        EXPERIMENTS_FOLDER / dataset_name / 'output' / f'iteration_{iteration}_top_k.tsv', header=True,
        sep='\t', index=False)
    # Compute and save per-user NDCG@k, matching RecBole's internal evaluation pool
    ndcg_per_user = get_recbole_ndcg_per_user(model, dataset, valid_data, config, k=k)
    ndcg_df = pd.DataFrame([{'user_id': uid, f'ndcg@{k}': val} for uid, val in ndcg_per_user.items()])
    ndcg_df.to_csv(
        EXPERIMENTS_FOLDER / dataset_name / 'output' / f'iteration_{iteration}_ndcg_per_user.tsv',
        sep='	', index=False
    )


    # Apply Choice model to select new recommendations from the top k
    accepted_df = accept_new_recommendations(choice_model, top_k_df, demographics_df, tracks_df)
    # Rename the columns to the correct names for recbole
    accepted_df = accepted_df[['user_id', 'item_id']].rename(
        columns={'user_id': 'user_id:token', 'item_id': 'item_id:token'})
    accepted_df.to_csv(
        EXPERIMENTS_FOLDER / dataset_name / 'output' / f'iteration_{iteration}_accepted_songs.tsv',
        header=True, sep='\t', index=False)

    # Append new interactions to dataset_df
    df_new = pd.concat([dataset_df.copy(), accepted_df], ignore_index=True)
    # Sort them again such that all interactions are grouped by user_id
    df_new = df_new.sort_values(['user_id:token', 'item_id:token']).reset_index(drop=True)
    df_new.to_csv(
        EXPERIMENTS_FOLDER / dataset_name / 'datasets' / f'iteration_{iteration + 1}.inter',
        header=True, sep='\t', index=False)

    # save embeddings of the current model 
    model_path = str(next(Path('saved').iterdir()))
    model = torch.load(model_path)
    user_embedding = model['state_dict']["user_embedding.weight"]
    item_embedding = model['state_dict']["item_embedding.weight"]

    torch.save(user_embedding, EXPERIMENTS_FOLDER / dataset_name / 'output' / f'iteration_{iteration}_user_embedding.pt')
    torch.save(item_embedding, EXPERIMENTS_FOLDER / dataset_name / 'output' / f'iteration_{iteration}_item_embedding.pt')


    # Copy checkpoint to output folder so it survives the next iteration's prepare_run (which clears saved/)
    shutil.copy(model_path, EXPERIMENTS_FOLDER / dataset_name / 'output' / f'iteration_{iteration}_checkpoint.pth')


    # Cleanup after finished loop
    cleanup(dataset_name, iteration)


if __name__ == '__main__':
    argh.dispatch_command(do_single_loop)


