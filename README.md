# VA4RecSys API

## Requirements

Find requirements in environment.yaml

## Getting Started

### Start API

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

## API Documentation

### Request

The API waits for a dictionary of the form:

```json
{
    "n": 2, 
    "dataset": "babyLFM5k", 
    "model": "BPR", 
    "choice_model": "consume_all", 
    "config": "recbole_config_default.yaml", 
    "artists_to_exclude": null 
}
```

**Parameters:**
- `n`: number iterations to run
- `dataset`: dataset (usually don't change that)
- `model`: RecBole model to use
- `choice_model`: choice model to use
- `config`: config file for additional training parameters
- `artists_to_exclude`: artists to exclude from recommendations

### Response

The API returns a dictionary of the form:

```json
{
    "dataset": "babyLFM5k",
    "results_folder": "experiments/babyLFM5k/results",
    "exists": true,
    "files": {
        "Embeddings": {
            "iteration_1_user_embedding.pt": {...},
            "iteration_1_item_embedding.pt": {...},
            "iteration_2_user_embedding.pt": {...}
        },
        "Accepted_songs": {
            "accepted_songs_iteration_1.csv": {...},
            "accepted_songs_iteration_2.csv": {...}
        },
        "Proportions": {
            "country_proportions_iteration_1.csv": {...}, #country proportions for each user (e.g. what percentage of male did user i consume in iteration 1)
            "gender_proportions_iteration_1.csv": {...}, #gender proportions for each user
            "country_proportions_iteration_2.csv": {...},
            "user_country_train_proportions.csv": {...}, 
            "user_gender_train_proportions.csv": {...}
        },
        "User": {
            "user_statistics.csv": {...}
        }
    }
}
```

**Response Fields:**
- `dataset`: the dataset name
- `results_folder`: path to the results folder
- `exists`: whether results folder exists
- `files`: nested structure of result files

#### File Entry Format

Each file entry contains:

```json
{
    "content_type": "text or base64",
    "content": "file contents as string"
}
```

- `content_type`: "text" for readable files, "base64" for binary files
- `content`: file contents as string
