import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch
from helper_files.metrics import average_pop

def evaluate(experiments_folder, experiment_name):

    # for RG1

    tracks = pd.read_csv(f'{experiments_folder}/{experiment_name}/input/tracks.tsv', sep="\t", header=None, names=["track_id","artist","title","country","gender","language","popularity_bin"])
    dataset = pd.read_csv(f'{experiments_folder}/{experiment_name}/input/dataset.inter', sep="\t", header=0)
    demographics = pd.read_csv(f'{experiments_folder}/{experiment_name}/input/demographics.tsv', sep="\t", header=None, names=["country","age","gender","user_id"])

    artist_exposures = tracks[["artist","country","gender","language","popularity_bin"]].drop_duplicates()

    artists_popularity = tracks.merge(dataset, left_on="track_id", right_on="item_id:token")["artist"].value_counts()/len(dataset)

    artist_exposures = artist_exposures.merge(artists_popularity, left_on="artist", right_index=True, how="left").rename(columns={"count":"train_popularity"})

    log_folders = [f for f in os.listdir(f'{experiments_folder}/{experiment_name}/log') if os.path.isdir(os.path.join(f'{experiments_folder}/{experiment_name}/log', f))]
    num_iter = max([int(f.split('_')[-1]) for f in log_folders])

    # for RG2.2
    user_statistics_all_iters = demographics.rename(columns={"country":"user_country","gender":"user_gender","age":"user_age"})

    for i in np.arange(1,num_iter+1):
        top_k = pd.read_csv(f"{experiments_folder}/{experiment_name}/output/iteration_{i}_top_k.tsv", sep="\t", header=0)
        top_k["exposure"] = 1/np.log2(1 + top_k["rank"])
        ndcg = pd.read_csv(f"{experiments_folder}/{experiment_name}/output/iteration_{i}_ndcg_per_user.tsv", sep="\t", header=0)

        iteration_exposure = tracks.merge(top_k, left_on="track_id", right_on="item_id")[["artist","exposure"]].groupby("artist").sum()
        iteration_exposure = iteration_exposure / iteration_exposure.sum()

        artist_exposures = artist_exposures.merge(iteration_exposure, left_on="artist", right_index=True, how="left").rename(columns={"exposure":f"iteration_{i}_exposure"}).fillna(0)

        # for RG2.1
        accepted_songs = top_k[["user_id","rank","exposure"]]
        accepted_songs["artist"] = top_k.merge(tracks[["track_id","artist"]], left_on="item_id", right_on="track_id")["artist"]
        accepted_songs = accepted_songs.merge(demographics, left_on="user_id", right_on="user_id", how="left")
        accepted_songs.rename(columns={"country":"user_country","gender":"user_gender","age":"user_age"}, inplace=True)

        accepted_songs = accepted_songs.merge(tracks[["artist","country","gender","language","popularity_bin"]].drop_duplicates(), left_on="artist", right_on="artist", how="left").rename(columns={"country":"artist_country","gender":"artist_gender"})

        accepted_songs.to_csv(f"{experiments_folder}/{experiment_name}/results/accepted_songs_iteration_{i}.csv", index=False)

        # for RG2.2 
        user_statistics_all_iters = user_statistics_all_iters.merge(ndcg, left_on="user_id", right_on="user_id", how="left").rename(columns={"ndcg@10":f"iteration_{i}_ndcg@10"})
        pop = average_pop(accepted_songs, artist_exposures)

        user_statistics_all_iters[f"iteration_{i}_average_pop"] = pop

        temp_country_proportions_df = pd.concat([
        accepted_songs[['user_id', 'artist_country']], 
        pd.get_dummies(accepted_songs['artist_country'].str.lstrip('_'), prefix='')
        ], axis=1)
        country_proportions_df = temp_country_proportions_df.groupby('user_id')[temp_country_proportions_df.columns.difference(['user_id', 'artist_country'])].mean().reset_index()
        country_proportions_df.to_csv(f"{experiments_folder}/{experiment_name}/results/country_proportions_iteration_{i}.csv", index=False)

        temp_gender_proportions_df = pd.concat([
        accepted_songs[['user_id', 'artist_gender']], 
        pd.get_dummies(accepted_songs['artist_gender'].str.lstrip('_'), prefix='')
        ], axis=1)
        gender_proportions_df = temp_gender_proportions_df.groupby('user_id')[temp_gender_proportions_df.columns.difference(['user_id', 'artist_gender'])].mean().reset_index()
        gender_proportions_df.to_csv(f"{experiments_folder}/{experiment_name}/results/gender_proportions_iteration_{i}.csv", index=False)

        # for RG3.1
        item_embedding = torch.load(f"{experiments_folder}/{experiment_name}/output/iteration_{i}_item_embedding.pt")
        user_embedding = torch.load(f"{experiments_folder}/{experiment_name}/output/iteration_{i}_user_embedding.pt")

        torch.save(item_embedding, f"{experiments_folder}/{experiment_name}/results/iteration_{i}_item_embedding.pt")
        torch.save(user_embedding, f"{experiments_folder}/{experiment_name}/results/iteration_{i}_user_embedding.pt")

    artist_exposures.to_csv(f"{experiments_folder}/{experiment_name}/results/artist_exposures.csv", index=False)

    # for RG2.2
    dataset_artists = tracks.merge(dataset, left_on="track_id", right_on="item_id:token")[["user_id:token","artist"]].rename(columns={"user_id:token":"user_id"})
    train_pop_per_user = dataset_artists.merge(artist_exposures[["artist","train_popularity"]], left_on="artist", right_on="artist", how="left")[["user_id","train_popularity"]].groupby("user_id").mean()
    user_statistics_all_iters["train_pop"] = train_pop_per_user

    user_statistics_all_iters.to_csv(f"{experiments_folder}/{experiment_name}/results/user_statistics.csv", index=False)

    #gender proportions
    temp_gender_proportions_df = dataset_artists.merge(artist_exposures[["artist","gender"]], left_on="artist", right_on="artist", how="left")
    for gender in temp_gender_proportions_df["gender"].unique():
        temp_gender_proportions_df[str(gender)] = temp_gender_proportions_df["gender"] == gender
    
    temp_gender_proportions_df = temp_gender_proportions_df.drop(columns=["artist"])
    gender_proportions_df = temp_gender_proportions_df.groupby('user_id').agg({
        'Male': 'mean',
        'Female': 'mean',
        'Non-binary': 'mean',
        'Other': 'mean'}).reset_index()
    
    gender_proportions_df.to_csv(f"{experiments_folder}/{experiment_name}/results/user_gender_train_proportions.csv", index=False)

    #country proportions
    temp_country_proportions_df = dataset_artists.merge(artist_exposures[["artist","country"]], left_on="artist", right_on="artist", how="left")

    country_proportions_df = pd.concat([
        temp_country_proportions_df[['user_id', 'country']], 
        pd.get_dummies(temp_country_proportions_df['country'].str.lstrip('_'), prefix='')
    ], axis=1)

    country_proportions_df = country_proportions_df.groupby('user_id')[country_proportions_df.columns.difference(['user_id', 'country'])].mean().reset_index()
    country_proportions_df.to_csv(f"{experiments_folder}/{experiment_name}/results/user_country_train_proportions.csv", index=False)









