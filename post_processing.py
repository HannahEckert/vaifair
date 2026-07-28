import argh

import pandas as pd
import numpy as np
from tqdm import tqdm
from argh import arg
import os

def prepare_post_processing(top_k, tracks_path, dataset_path):

    #top_k = pd.read_csv(top_k_path, sep='\t')
    tracks = pd.read_csv(tracks_path, sep="\t", header=None, names=["track_id","artist","title","country","gender","language","popularity_bin"])
    dataset = pd.read_csv(dataset_path, sep="\t", header=0)

    #normalize scores by user
    top_k['normalized_score'] = top_k.groupby('user_id')['score'].transform(lambda x: x / x.sum())

    

    item_artist_matching = tracks[["track_id", "artist"]].drop_duplicates()
    artist_country_gender = tracks[["artist", "country", "gender", "language", "popularity_bin"]].drop_duplicates(subset="artist")

    top_k = top_k.merge(item_artist_matching, left_on="item_id", right_on="track_id", how="left")

    top_k = top_k.merge(artist_country_gender, left_on="artist", right_on="artist", how="left")

    dataset = dataset.merge(item_artist_matching, left_on="item_id:token", right_on="track_id", how="left")
    dataset = dataset.merge(artist_country_gender, left_on="artist", right_on="artist", how="left")

    return tracks, top_k, dataset


def baseline_marras(tracks, top_k, dataset, l=0.25, target_distribution = "interactions", dimension="country"):
    #dimension can be "country" or "gender"

    #create target distributions
    if target_distribution == "interactions":
        target_distribution = dataset.groupby([dimension]).size() / dataset.shape[0]
    elif target_distribution == "catalog":
        target_distribution = tracks.groupby([dimension]).size() / tracks.shape[0]
    else:
        raise ValueError("Invalid target distribution. Choose 'interactions' or 'catalog'.")


    # Build a group index mapping -> integer position in target array
    group_index = {key: i for i, key in enumerate(target_distribution.index)}
    G = len(target_distribution)
    sqrt_target = np.sqrt(target_distribution.values)  # shape (G,) — computed once

    results = []

    for user in tqdm(top_k["user_id"].unique(), desc="Re-ranking users"):
        user_top_k = top_k[top_k["user_id"] == user].reset_index(drop=True)

        # Map each candidate to its group index (-1 if unknown)
        candidate_groups = np.array([
            group_index.get(row[dimension], -1)
            for _, row in user_top_k.iterrows()
        ], dtype=np.int32)
        relevance = user_top_k["normalized_score"].to_numpy()

        # Cumulative discounted exposure vector over groups
        current_exposure = np.zeros(G, dtype=np.float64)
        selected_mask = np.zeros(len(user_top_k), dtype=bool)

        for position in range(1, 11):
            discount = 1.0 / np.log2(position + 1)
            candidate_mask = ~selected_mask
            if not candidate_mask.any():
                break

            # --- Vectorized Hellinger over all candidates at once ---
            # hyp_exposure[c] = current_exposure + discount * one_hot(group[c])
            # shape: (N_cands, G)
            hyp_exposure = np.tile(current_exposure, (candidate_mask.sum(), 1))
            cand_indices = np.where(candidate_mask)[0]
            cand_groups = candidate_groups[cand_indices]

            # Add discount only for candidates with a known group
            known = cand_groups >= 0
            hyp_exposure[known, cand_groups[known]] += discount

            totals = hyp_exposure.sum(axis=1, keepdims=True)
            totals[totals == 0] = 1.0  # avoid division by zero
            hyp_dist = hyp_exposure / totals  # (N_cands, G)

            # Squared Hellinger: 0.5 * sum((sqrt(target) - sqrt(hyp))^2, axis=1)
            hellinger_sq = 0.5 * np.sum((sqrt_target - np.sqrt(hyp_dist)) ** 2, axis=1)

            scores = (1 - l) * relevance[cand_indices] - l * hellinger_sq
            best_local = np.argmax(scores)
            best_idx = cand_indices[best_local]

            # Commit chosen item
            selected_mask[best_idx] = True
            g = candidate_groups[best_idx]
            if g >= 0:
                current_exposure[g] += discount

            chosen = user_top_k.iloc[best_idx].copy()
            chosen["rank"] = position
            results.append(chosen)

    results = pd.DataFrame(results)
    return results



def mitigation_continent(tracks, top_k, dataset, l=1.0, target_distribution="interactions", dimension="country",
                          reranking_type="exposure", k=10):
    """
    Implementation of the mitigationContinent algorithm (Deldjoo et al.), adapted to this
    project's setting to match baseline_marras()'s interface: a single `dimension` ("country"
    or "gender") instead of intersectional (country, gender) groups, and target distributions
    computed from `dataset` (interactions) or `tracks` (catalog) — same as baseline_marras.

    :param tracks: tracks DataFrame (as returned by prepare_post_processing())
    :param top_k: candidate DataFrame with up to ~100 ranked candidates per user (as produced
                  by compute_top_k_scores() and prepare_post_processing()), already merged
                  with tracks metadata so it has a `dimension` and 'normalized_score' column
    :param dataset: interactions DataFrame (as returned by prepare_post_processing(), already
                    merged with tracks metadata)
    :param l: fraction of possible swaps to consider at most (0 < l <= 1.0).
              e.g. l=0.1 means at most 10% of candidate swaps are applied.
    :param target_distribution: "interactions" (from dataset) or "catalog" (from tracks)
    :param dimension: dimension to consider for re-ranking ("country" or "gender")
    :param reranking_type: "visibility" (count-based) or "exposure" (discounted-exposure-based)
    :param k: top-k cutoff (default 10)
    """

    ## I think there are bugs!!!!!!!!!!!!!###############

    def get_group(row):
        """Return this row's value for `dimension`, or None if missing."""
        val = row.get(dimension)
        return None if pd.isna(val) else val

    # ── Target proportions (same semantics as baseline_marras) ────────────────
    if target_distribution == "interactions":
        target_props = dataset.groupby([dimension]).size() / dataset.shape[0]
    elif target_distribution == "catalog":
        target_props = tracks.groupby([dimension]).size() / tracks.shape[0]
    else:
        raise ValueError("Invalid target distribution. Choose 'interactions' or 'catalog'.")

    # ── Initial proportions from the current top-k ───────────────────────────
    top_k_df = top_k[top_k["rank"] <= k].copy()
    top_k_df["exposure"] = 1 / np.log2(top_k_df["rank"] + 1)
    total_exposure   = top_k_df["exposure"].sum()
    total_visibility = float(len(top_k_df))

    top_k_df["_group"] = top_k_df.apply(get_group, axis=1)
    valid_top_k = top_k_df.dropna(subset=["_group"])

    if reranking_type == "visibility":
        proportions = valid_top_k.groupby("_group").size().astype(float) / total_visibility
    else:
        proportions = valid_top_k.groupby("_group")["exposure"].sum() / total_exposure

    # Align to full group index
    all_groups   = proportions.index.union(target_props.index)
    target_props = target_props.reindex(all_groups, fill_value=0)
    proportions  = proportions.reindex(all_groups, fill_value=0)
    # group_balance[g] > 0 → over-represented (advantaged); < 0 → under-represented (disadvantaged)
    group_balance = proportions - target_props

    # ── Working copy ──────────────────────────────────────────────────────────
    working = top_k.copy()
    working["exposure"] = 1 / np.log2(working["rank"] + 1)
    working["_group"]   = working.apply(get_group, axis=1)

    # ── Collect candidate swaps across all users ──────────────────────────────
    possible_swaps = []

    for user_id, user_df in tqdm(working.groupby("user_id"), desc="Collecting swaps"):
        user_df      = user_df.sort_values("rank")
        user_top_k   = user_df[user_df["rank"] <= k].sort_values("rank", ascending=False)
        user_outside = user_df[user_df["rank"] > k].sort_values("rank")

        out_cands, in_cands = [], []

        for _, row in user_top_k.iterrows():
            grp = row["_group"]
            if grp is None:
                continue
            if group_balance.get(grp, 0) > 0:   # advantaged → candidate to remove
                out_cands.append(row)

        for _, row in user_outside.iterrows():
            grp = row["_group"]
            if grp is None:
                continue
            if group_balance.get(grp, 0) < 0:   # disadvantaged → candidate to insert
                in_cands.append(row)

        i_in, i_out = 0, 0
        while i_in < len(in_cands) and i_out < len(out_cands):
            item_in  = in_cands[i_in]
            item_out = out_cands[i_out]
            loss = item_out["normalized_score"] - item_in["normalized_score"]
            possible_swaps.append({
                "user_id":  user_id,
                "idx_out":  item_out.name,
                "idx_in":   item_in.name,
                "grp_out":  item_out["_group"],
                "grp_in":   item_in["_group"],
                "rank_out": item_out["rank"],
                "rank_in":  item_in["rank"],
                "exp_out":  item_out["exposure"],
                "exp_in":   item_in["exposure"],
                "loss":     loss,
            })
            i_in  += 1
            i_out += 1

    # Sort by loss ascending (minor loss first)
    possible_swaps.sort(key=lambda x: x["loss"])

    # Apply at most l-fraction of the candidate swaps
    max_swaps = max(1, int(len(possible_swaps) * l))
    possible_swaps = possible_swaps[:max_swaps]

    # ── Apply swaps greedily ───────────────────────────────────────────────────
    used_idx     = set()
    rank_updates = {}  # original df index → new rank

    for swap in possible_swaps:
        idx_out = swap["idx_out"]
        idx_in  = swap["idx_in"]

        if idx_out in used_idx or idx_in in used_idx:
            continue

        grp_out = swap["grp_out"]
        grp_in  = swap["grp_in"]

        # Re-check conditions with updated group_balance
        if group_balance.get(grp_out, 0) <= 0:
            continue   # no longer advantaged
        if group_balance.get(grp_in, 0) >= 0:
            continue   # no longer disadvantaged

        # Commit swap: exchange ranks
        rank_updates[idx_out] = swap["rank_in"]
        rank_updates[idx_in]  = swap["rank_out"]

        # Update proportions
        if reranking_type == "visibility":
            exp_diff = 1.0 / total_visibility
        else:
            exp_diff = (swap["exp_out"] - swap["exp_in"]) / total_exposure

        proportions[grp_out] -= exp_diff
        proportions[grp_in]  += exp_diff
        group_balance = proportions - target_props

        used_idx.add(idx_out)
        used_idx.add(idx_in)

    # ── Apply rank updates and return top-k ───────────────────────────────────
    working["rank"] = working.apply(lambda row: rank_updates.get(row.name, row["rank"]), axis=1)
    result = working[working["rank"] <= k].copy().drop(columns=["exposure", "_group"], errors="ignore")
    return result



def post_processing(top_k, dataset, dimension, l, target_distribution, seed):


    tracks_path = f"experiments/{dataset}/input/tracks.tsv"
    dataset_path = f"experiments/{dataset}/input/dataset_filtered.inter"

    tracks, top_k, dataset = prepare_post_processing(top_k, tracks_path, dataset_path)

    # Shuffle user order with the given seed
    rng = np.random.default_rng(seed)
    user_order = top_k["user_id"].unique()
    user_order = rng.permutation(user_order)
    top_k = pd.concat([top_k[top_k["user_id"] == u] for u in user_order], ignore_index=True)

    results = baseline_marras(tracks, top_k, dataset, l=l, target_distribution=target_distribution, dimension=dimension)
    #results = mitigation_continent(tracks, top_k, dataset, l=l, target_distribution=target_distribution, dimension=dimension, reranking_type="exposure", k=10)

    return results


"""

@arg("--dataset", type = str, default = "babyLFM5k")
@arg("--iter" , type=int, default=1)
@arg("--dimension", type=str, default="country", help="Dimension to consider for re-ranking (country or gender)")
@arg('--l', type=float, default=0.25, help='Trade-off parameter')
@arg("--target_distribution", type=str,default = "interactions" )
@arg("--seed", type=int, default=42, help='Random seed for shuffling user order')

def post_processing(
    *,
    dataset="babyLFM5k",
    iter=1,
    l=0.25,
    target_distribution="interactions",
    dimension="country",
    seed=42
):
    top_k_path = f"experiments/{dataset}/output/iteration_{iter}_top_100.tsv"
    tracks_path = f"experiments/{dataset}/input/tracks.tsv"
    dataset_path = f"experiments/{dataset}/input/dataset_filtered.inter"
    output_path = f"experiments/{dataset}/output/iteration_{iter}_top_k.tsv"
    output_path2 = f"experiments/{dataset}/output/iteration_{iter}_accepted_songs.tsv"

    tracks, top_k, dataset = prepare_post_processing(top_k_path, tracks_path, dataset_path)

    # Shuffle user order with the given seed
    rng = np.random.default_rng(seed)
    user_order = top_k["user_id"].unique()
    user_order = rng.permutation(user_order)
    top_k = pd.concat([top_k[top_k["user_id"] == u] for u in user_order], ignore_index=True)

    results = baseline_marras(tracks, top_k, dataset, l=l, target_distribution=target_distribution, dimension=dimension)
        
    
    #create output folder if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)
    results.to_csv(output_path2, index=False)



if __name__ == '__main__':
    argh.dispatch_command(post_processing)

"""

    

