import shutil
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_run_dir(runs_dir: str, experiment_name: str, config_path: str) -> Path:
    run_dir = Path(runs_dir) / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, run_dir / "config.yaml")
    return run_dir
