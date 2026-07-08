import numpy as np
import pandas as pd

def average_pop(accepted_songs, artist_exposures):
    average_pop = accepted_songs[["user_id","artist"]].merge(artist_exposures[["artist","train_popularity"]], on="artist", how="left")[["user_id","train_popularity"]].groupby("user_id").mean().rename(columns={"train_popularity":"mean_train_popularity"})
    return average_pop