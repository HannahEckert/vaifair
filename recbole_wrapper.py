import math
import sys

import numpy as np
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

def get_recbole_ndcg_per_user(model, dataset, valid_data, config: Config, k: int = 10, batch_size: int = 32):
    """
    Computes per-user NDCG@k exactly as RecBole's internal evaluator does.

    full_sort_scores must be called with valid_data (not test_data) so that
    RecBole only masks training items internally — leaving validation items
    available for ranking, just like the internal validation evaluator does.

    Returns a dict mapping original user_id -> ndcg@k value.
    """
    user_ids, item_ids = _get_ids(dataset)

    scores = np.empty((len(user_ids), len(item_ids)), dtype=np.float32)

    for i in trange(math.ceil(len(user_ids) / batch_size), desc=f'Calculating NDCG scores',
                    dynamic_ncols=True, smoothing=0):
        start = i * batch_size
        end = min(len(user_ids), (i + 1) * batch_size)
        # Use valid_data so RecBole only masks training items before scoring,
        # matching its internal validation evaluation pool.
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

    # Build reverse mappings: RecBole internal ID -> original ID
    recbole_to_orig_user = {v: int(key) for key, v in dataset.field2token_id['user_id'].items() if key != '[PAD]'}
    recbole_to_orig_item = {v: int(key) for key, v in dataset.field2token_id['item_id'].items() if key != '[PAD]'}

    ndcg_per_user = {}
    for i, valid_items_tensor in enumerate(valid_data.uid2positive_item[1:]):
        recbole_uid = i + 1  # uid2positive_item[0] is [PAD]
        orig_uid = recbole_to_orig_user[recbole_uid]

        valid_items_orig = set(recbole_to_orig_item[iid.item()] for iid in valid_items_tensor)
        if not valid_items_orig:
            continue

        top_k_items = np.argsort(-scores[i])[:k]

        dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(top_k_items) if item in valid_items_orig)
        ideal_len = min(len(valid_items_orig), k)
        idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_len))
        ndcg_per_user[orig_uid] = dcg / idcg if idcg > 0 else 0.0

    return ndcg_per_user