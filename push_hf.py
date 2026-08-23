"""Create/update the Hugging Face Space. Run after: hf auth login"""
from pathlib import Path

from huggingface_hub import HfApi, create_repo, whoami

ROOT = Path(__file__).resolve().parent
user = whoami()["name"]
repo_id = f"{user}/flow-b-dashboard"

create_repo(
    repo_id,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
    private=False,
)

api = HfApi()
api.upload_folder(
    folder_path=str(ROOT),
    repo_id=repo_id,
    repo_type="space",
    ignore_patterns=[
        ".git",
        ".git/**",
        "__pycache__",
        "__pycache__/**",
        ".venv",
        ".venv/**",
        "venv",
        "venv/**",
        "*.pyc",
        "flow_b_cache/prepared_dashboard.pkl",
        "flow_b_cache/prepared_dashboard.meta.json",
    ],
)
print(f"Space: https://huggingface.co/spaces/{repo_id}")
