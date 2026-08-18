# VAIFAiR Backend

This repository contains the source code for the VAIFAiR backend. The instructions below describe how to run the backend as an API. To run the simulation directly from Python instead, execute run_loop.py using the same arguments as those provided in the POST request.

## Requirements

conda create -n test_env python=3.10 -y

conda activate test_env

pip install -r requirements.txt


## Getting Started

### Start API

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

## API Documentation

### POST `/runs` - Start a Run

Send a POST request with parameters:

```json
{
    "n": 1,
    "dataset": "babyLFM5k",
    "model": "BPR",
    "choice_model": "consume_all",
    "config": "recbole_config_default.yaml",
    "artists_to_exclude": null,
    "pp": false,
    "pp_dimension": "country",
    "pp_l": 0.25,
    "pp_target_distribution": "interactions",
    "pp_seed": 42
}
```
`n` are the number of iterations in the simulation

`dataset` is the dataset used for the simulation run. Give new dataset in the experiments folder.

`choice_model` ist the user choice model. Can be `consume_all` or `random`

`config` is the recbole config file where additional training details "such as embedding size, batch size, .. ) can be fixed. Give new config file directly in teh main folder

`artist_to_exclude` give a list of artists you want to exclude in the simulation

`pp` enables fairness-aware post-processing re-ranking. 

`pp_dimension` is `country`,  `gender`,  `language` or  `popularity_bin`;

`pp_l` is the trade-off parameter between relevance and fairness; 

`pp_target_distribution` is `interactions` or `catalog`; 

`pp_seed` seeds the user-order shuffle.

Returns immediately with a job ID:

```json
{
    "job_id": "uuid-string",
    "status": "queued",
    "status_url": "/runs/uuid-string"
}
```

### GET `/runs/{job_id}` - Check Status

Poll this endpoint to check progress. Returns current `status` ("queued", "running", "completed", or "failed"), logs, and results when done.

### Health Check

`GET /` returns `{"status": "ok", "message": "..."}`

### Example

Start a run with curl:

```bash
curl -X POST https://vista-be.test.cp.jku.at/runs \
  -H "Content-Type: application/json" \
  -d '{
    "n": 1,
    "dataset": "babyLFM5k",
    "model": "BPR",
    "choice_model": "consume_all",
    "config": "recbole_config_default.yaml"
  }'
```

Check status:

```bash
curl https://vista-be.test.cp.jku.at/runs/{job_id}
```

Replace `{job_id}` with the ID from the POST response.

# Additional Training Details 

The simulations used in both the usage scenario and the expert study of the paper were conducted using the default configuration of the hosted interface. These default settings are as follows. 
The recommender system uses a matrix factorization architecture trained with BPR loss and an embedding size of 256. Each simulation spans 15 iterations, with the recommender system trained for 20 epochs per iteration and initialized from the weights of the previous iteration. The training and evaluation batch sizes are both set to 2048.
A random seed of 42 was used to ensure reproducibility.
The recommender system is trained on a subset of the LFM-2b dataset. We construct this subset by restricting the data to interactions from 2019 and filtering users and artists based on the availability of the attributes used in our analysis. We then randomly sample 5,000 artists and retain the corresponding songs, users, and interactions to form the final dataset. 
