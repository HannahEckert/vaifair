import math
import sys

import numpy as np
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation, construct_transform
from recbole.utils import init_seed, init_logger, get_model, get_flops, set_color, get_trainer, get_environment
from logging import getLogger

from recbole.utils.case_study import full_sort_scores
from tqdm import trange

import os

def run_recbole_experiment(model: str, dataset: str, config: Config, checkpoint_path: str = None):
    """
    Initially we used recbole.quick_start.run_recbole() to run the RecBole models.
    However, this has many limitations and undesired behaviour and thus we implemented the function ourselves

    :param checkpoint_path: Optional path to a previously saved .pth checkpoint. If provided, the model weights
                            are loaded from this checkpoint before training (train-from-checkpoint / continued training).
    """
    init_seed(config["seed"], config["reproducibility"])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # initialize the dataset according to config
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting. Test_data is always empty and thus ignored in our case
    logger.info('Preparing dataset')
    train_data, valid_data, test_data = data_preparation(config, dataset)
    logger.info('Done!')

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model_class = get_model(config["model"])
    # instantiate the model
    model = model_class(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config["device"])
        # Load state dict into model
        #if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            #model.load_state_dict(checkpoint['state_dict'])
        #else:
            #model.load_state_dict(checkpoint)
        logger.info("Checkpoint loaded successfully")

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")

    saved_dir = config["checkpoint_dir"]
    if checkpoint_path is not None and not any(os.scandir(saved_dir)):
        fallback_path = os.path.join(saved_dir, 'latest_checkpoint.pth')
        trainer._save_checkpoint(epoch=trainer.epochs, saved_model_file=fallback_path)

    # cleanup to hopefully avoid memory leaks
    del model
    del trainer
    del train_data
    del valid_data
    del test_data


def _get_ids(dataset):
    """Recbole internally uses different IDs to ours, and this mapping allows us to properly process their recommendations"""
    user_ids = list(dataset.field2token_id['user_id'].keys())
    # [PAD] user
    user_ids.remove('[PAD]')
    user_ids = dataset.token2id(dataset.uid_field, user_ids)
    user_ids = user_ids.astype(np.int64)

    item_ids = list(dataset.field2token_id['item_id'].keys())
    # [PAD] item
    item_ids.remove('[PAD]')
    item_ids = dataset.token2id(dataset.iid_field, item_ids)
    item_ids = item_ids.astype(np.int64)

    return user_ids, item_ids

def get_recbole_scores(model, dataset, test_data, config: Config, batch_size: int = 32):
    """Calculates the scores for all items and users"""
    user_ids, item_ids = _get_ids(dataset)

    scores = np.empty((len(user_ids), len(item_ids)), dtype=np.float32)

    for i in trange(math.ceil(len(user_ids) / batch_size), desc=f'Calculating Recommendation Scores',
                    dynamic_ncols=True, smoothing=0):
        start = i * batch_size
        end = min(len(user_ids), (i + 1) * batch_size)

        batch_scores = full_sort_scores(user_ids[start:end], model, test_data,
                                        device=config['device']).cpu().numpy().astype(
            np.float32)
        scores[start:end] = batch_scores[:, item_ids]

    # set scores of test set items to -inf such that they are never recommended
    for i, items in enumerate(test_data.uid2positive_item[1:]):
        # -1 because RecBole uses 1-based indexing with a [PAD] item
        items = items.cpu().numpy() - 1
        scores[i, items] = -np.inf

    # Recbole uses its own IDs internally, but before continuing we need to map them back to the original Item IDs
    item_id_mapping = np.zeros(len(item_ids), dtype=np.int64)
    for i in range(0, len(item_ids)):
        try:
            # Gets the internal ID within the scores
            recbole_column_index = dataset.field2token_id['item_id'][str(i)]
            # Recboles Indices start at 1 because they have a [PAD] item at 0
            item_id_mapping[i] = recbole_column_index - 1
        except KeyError:
            # This item was not present in the training data and thus has no internal ID. We will just set it to -1 and ignore it later
            item_id_mapping[i] = -1

    # Get a new view to scores_full with the columns in the correct order
    scores = scores[:, item_id_mapping]

    return scores

def get_recbole_eval_scores(model, dataset, valid_data, config: Config, batch_size: int = 32):
    """
    Computes scores for all users/items using valid_data for RecBole's internal masking,
    i.e. only training interactions are masked out — validation items remain visible/scoreable.

    Use this (instead of get_recbole_scores(), which masks with test_data and therefore
    excludes validation items entirely) whenever validation items need to be rankable —
    e.g. to build a top-k ranking for evaluating NDCG@k against the validation set, with or
    without post-processing/re-ranking applied afterwards.
    """
    user_ids, item_ids = _get_ids(dataset)

    scores = np.empty((len(user_ids), len(item_ids)), dtype=np.float32)

    for i in trange(math.ceil(len(user_ids) / batch_size), desc=f'Calculating Evaluation Scores',
                    dynamic_ncols=True, smoothing=0):
        start = i * batch_size
        end = min(len(user_ids), (i + 1) * batch_size)
        # Use valid_data so RecBole only masks training items before scoring,
        # leaving validation items available for ranking.
        batch_scores = full_sort_scores(user_ids[start:end], model, valid_data,
                                        device=config['device']).cpu().numpy().astype(np.float32)
        scores[start:end] = batch_scores[:, item_ids]

    # Map RecBole internal item indices back to original item IDs
    item_id_mapping = np.zeros(len(item_ids), dtype=np.int64)
    for i in range(len(item_ids)):
        try:
            recbole_column_index = dataset.field2token_id['item_id'][str(i)]
            item_id_mapping[i] = recbole_column_index - 1
        except KeyError:
            item_id_mapping[i] = -1  # Item not present in training data
    scores = scores[:, item_id_mapping]

    return scores


def get_recbole_ndcg_per_user(model, dataset, valid_data, config: Config, k: int = 10, batch_size: int = 32):
    """
    Computes per-user NDCG@k exactly as RecBole's internal evaluator does, ranking directly
    by the model's raw scores (no post-processing/re-ranking applied).

    Returns a dict mapping original user_id -> ndcg@k value.
    """
    scores = get_recbole_eval_scores(model, dataset, valid_data, config, batch_size)

    # Row i in `scores` corresponds to RecBole internal user id i + 1 (uid2positive_item[0] is [PAD])
    recbole_to_orig_user = {v: int(key) for key, v in dataset.field2token_id['user_id'].items() if key != '[PAD]'}
    valid_items_per_user = _get_valid_items_per_user(dataset, valid_data)

    ndcg_per_user = {}
    for i, valid_items_tensor in enumerate(valid_data.uid2positive_item[1:]):
        recbole_uid = i + 1
        orig_uid = recbole_to_orig_user[recbole_uid]

        valid_items_orig = valid_items_per_user[orig_uid]
        if not valid_items_orig:
            continue

        top_k_items = np.argsort(-scores[i])[:k]
        ndcg_per_user[orig_uid] = _compute_ndcg(top_k_items, valid_items_orig, k)

    return ndcg_per_user


def _get_valid_items_per_user(dataset, valid_data):
    """
    Builds a dict mapping original user_id -> set of original item_ids in the
    validation set (RecBole's ground truth positive items for that user).
    """
    recbole_to_orig_user = {v: int(key) for key, v in dataset.field2token_id['user_id'].items() if key != '[PAD]'}
    recbole_to_orig_item = {v: int(key) for key, v in dataset.field2token_id['item_id'].items() if key != '[PAD]'}

    valid_items_per_user = {}
    for i, valid_items_tensor in enumerate(valid_data.uid2positive_item[1:]):
        recbole_uid = i + 1  # uid2positive_item[0] is [PAD]
        orig_uid = recbole_to_orig_user[recbole_uid]
        valid_items_per_user[orig_uid] = set(recbole_to_orig_item[iid.item()] for iid in valid_items_tensor)

    return valid_items_per_user


def _compute_ndcg(top_k_items, valid_items_orig, k):
    """Computes NDCG@k for a single user given their ranked top-k items (original IDs, ordered by rank)
    and the set of ground-truth original item IDs."""
    dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(top_k_items[:k]) if item in valid_items_orig)
    ideal_len = min(len(valid_items_orig), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_len))
    return dcg / idcg if idcg > 0 else 0.0


def get_ndcg_from_top_k_df(top_k_df, dataset, valid_data, k: int = 10):
    """
    Computes per-user NDCG@k from an already-computed, ranked `top_k_df` instead of
    recomputing the ranking from raw model scores.

    IMPORTANT: `top_k_df` must be built from scores obtained via get_recbole_eval_scores()
    (masked with valid_data, so validation items remain visible/rankable) — NOT from
    get_recbole_scores() (masked with test_data), which excludes validation items entirely
    and would make NDCG always 0. This should be used instead of get_recbole_ndcg_per_user()
    whenever the final ranking differs from a plain argsort of those eval scores (e.g. when
    post-processing re-ranks the candidates), so that NDCG reflects what was actually ranked.

    :param top_k_df: DataFrame with columns 'user_id', 'item_id' and 'rank' (original IDs,
                      one row per recommended item, 'rank' starting at 1 = best), built from
                      get_recbole_eval_scores()-derived candidates.
    :param dataset: RecBole dataset object (used to map validation items back to original IDs).
    :param valid_data: RecBole validation data (used as ground truth), matching the pool used
                        by get_recbole_eval_scores().
    :param k: number of top items to consider per user.
    :returns: dict mapping original user_id -> ndcg@k value.
    """
    valid_items_per_user = _get_valid_items_per_user(dataset, valid_data)

    # Build ranked item lists per user from top_k_df, ordered by rank
    top_k_df = top_k_df.sort_values(['user_id', 'rank'])
    ranked_items_per_user = top_k_df.groupby('user_id')['item_id'].apply(list).to_dict()

    ndcg_per_user = {}
    for orig_uid, valid_items_orig in valid_items_per_user.items():
        if not valid_items_orig:
            continue

        top_k_items = ranked_items_per_user.get(orig_uid, [])
        ndcg_per_user[orig_uid] = _compute_ndcg(top_k_items, valid_items_orig, k)

    return ndcg_per_user

