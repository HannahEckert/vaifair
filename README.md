find requirements in environment.yaml

start api via:
python -m uvicorn api:app --host 0.0.0.0 --port 8000

waits for dic of the form:
{
    "n": 2, #number iterations to run
    "dataset": "babyLFM5k", #dataset usually dont change that 
    "model": "BPR", 
    "choice_model": "consume_all", 
    "config": "recbole_config_default.yaml", #config file for additional training parameters
    "artists_to_exclude": null 
  }
