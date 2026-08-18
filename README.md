# VA4RecSys API

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
