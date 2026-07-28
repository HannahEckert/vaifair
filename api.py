
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from run_loop import collect_results_as_dict

from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(
    title="Recommendation Experiment API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://vista.test.cp.jku.at"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


PROJECT_DIR = Path(__file__).resolve().parent


jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


class RunRequest(BaseModel):
    n: int = Field(
        default=3,
        ge=1,
        le=1000,
        description="Number of iterations",
    )

    dataset: str = Field(
        default="BabyLFM5k",
        min_length=1,
        max_length=100,
    )

    model: str = Field(
        default="BPR",
        min_length=1,
        max_length=100,
    )

    choice_model: str = Field(
        default="consume_all",
        min_length=1,
        max_length=100,
    )

    config: str = Field(
        default="recbole_config_default.yaml",
        min_length=1,
        max_length=255,
    )

    artists_to_exclude: list[str] | None = None

    pp: bool = Field(
        default=False,
        description="Whether to apply fairness-aware post-processing re-ranking",
    )

    pp_dimension: str = Field(
        default="country",
        description="Dimension to consider for re-ranking ('country' or 'gender')",
    )

    pp_l: float = Field(
        default=0.25,
        description="Trade-off parameter between relevance and fairness",
    )

    pp_target_distribution: str = Field(
        default="interactions",
        description="Target distribution for post-processing ('interactions' or 'catalog')",
    )

    pp_seed: int = Field(
        default=42,
        description="Random seed for shuffling user order in post-processing",
    )


class RunStartedResponse(BaseModel):
    job_id: str
    status: Literal["queued"]
    status_url: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_dataset_name(dataset: str) -> None:
    """
    Prevent values such as '../../some-folder'.

    Adjust this rule if your dataset folder names contain other characters.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", dataset):
        raise HTTPException(
            status_code=422,
            detail=(
                "Invalid dataset name. Use only letters, numbers, "
                "underscores, hyphens, and periods."
            ),
        )


def validate_config_path(config: str) -> None:
    """
    Ensure that the config file is located inside the project directory.
    """
    config_path = (PROJECT_DIR / config).resolve()

    try:
        config_path.relative_to(PROJECT_DIR)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="The config file must be inside the project directory.",
        ) from exc

    if not config_path.is_file():
        raise HTTPException(
            status_code=422,
            detail=f"Config file does not exist: {config}",
        )


def execute_run(job_id: str, request: RunRequest) -> None:
    """
    Run run_loop.py in a separate process.

    This function executes in a background thread so that the initial HTTP
    request can return immediately.
    """
    with jobs_lock:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["started_at"] = utc_now()

    command = [
        sys.executable,
        str(PROJECT_DIR / "run_loop.py"),
        "-n",
        str(request.n),
        "--dataset",
        request.dataset,
        "--model",
        request.model,
        "--choice-model",
        request.choice_model,
        "--config",
        request.config,
    ]

    if request.artists_to_exclude:
        command.append("--artists-to-exclude")
        command.extend(request.artists_to_exclude)

    if request.pp:
        command.append("--PP")
        command.extend(["--PP-dimension", request.pp_dimension])
        command.extend(["--PP-l", str(request.pp_l)])
        command.extend(["--PP-target-distribution", request.pp_target_distribution])
        command.extend(["--PP-seed", str(request.pp_seed)])

    try:
        completed_process = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )

        if completed_process.returncode != 0:
            with jobs_lock:
                jobs[job_id].update(
                    {
                        "status": "failed",
                        "finished_at": utc_now(),
                        "return_code": completed_process.returncode,
                        "stdout": completed_process.stdout,
                        "stderr": completed_process.stderr,
                        "error": "run_loop.py returned a non-zero exit code.",
                    }
                )
            return

        results = collect_results_as_dict(request.dataset)

        with jobs_lock:
            jobs[job_id].update(
                {
                    "status": "completed",
                    "finished_at": utc_now(),
                    "return_code": completed_process.returncode,
                    "stdout": completed_process.stdout,
                    "stderr": completed_process.stderr,
                    "results": results,
                }
            )

    except Exception as exc:
        with jobs_lock:
            jobs[job_id].update(
                {
                    "status": "failed",
                    "finished_at": utc_now(),
                    "error": str(exc),
                }
            )


@app.get("/")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "Recommendation Experiment API is running",
    }


@app.post(
    "/runs",
    response_model=RunStartedResponse,
    status_code=202,
)
def start_run(request: RunRequest) -> RunStartedResponse:
    """
    Start a new recommendation experiment.

    The endpoint immediately returns a job ID. The frontend can use that job
    ID to check whether the experiment has finished.
    """
    validate_dataset_name(request.dataset)
    validate_config_path(request.config)

    run_loop_path = PROJECT_DIR / "run_loop.py"

    if not run_loop_path.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Could not find {run_loop_path}",
        )

    job_id = str(uuid.uuid4())

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": utc_now(),
            "parameters": request.model_dump(),
        }

    worker = threading.Thread(
        target=execute_run,
        args=(job_id, request),
        daemon=True,
    )
    worker.start()

    return RunStartedResponse(
        job_id=job_id,
        status="queued",
        status_url=f"/runs/{job_id}",
    )


@app.get("/runs/{job_id}")
def get_run(job_id: str) -> dict:
    """
    Return the current status, logs, and results for a run.
    """
    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Job not found.",
            )

        return dict(job)


# Mount the experiments folder as static files
# This allows the frontend to access files from /experiments/* endpoints
experiments_dir = PROJECT_DIR / "experiments"
if experiments_dir.exists():
    app.mount("/experiments", StaticFiles(directory=str(experiments_dir)), name="experiments")
