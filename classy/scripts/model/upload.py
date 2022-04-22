import argparse
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import huggingface_hub

from classy.scripts.model.download import CLASSY_DATE_FORMAT, get_md5
from classy.scripts.model.export import zip_run
from classy.utils.experiment import Experiment

logger = logging.getLogger(__name__)


def create_info_file(tmpdir: Path):
    logger.debug("Computing md5 of model.zip")
    md5 = get_md5(tmpdir / "model.zip")
    date = datetime.now().strftime(CLASSY_DATE_FORMAT)

    logger.debug("Dumping info.json file")
    with (tmpdir / "info.json").open("w") as f:
        json.dump(dict(md5=md5, upload_date=date), f, indent=2)


def upload(
    model_name,
    organization: Optional[str] = None,
    repo_name: Optional[str] = None,
    commit: Optional[str] = None,
):
    token = huggingface_hub.HfFolder.get_token()
    if token is None:
        print(
            "No HuggingFace token found. You need to execute `huggingface-cli login` first!"
        )
        return

    exp = Experiment.from_name(model_name)
    if exp is None:
        print(f"No experiment named {model_name} found. Exiting...")
        return

    run = exp.last_valid_run
    if run is None:
        print(f"No valid run found for experiment {model_name}. Exiting...")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        api = huggingface_hub.hf_api.HfApi()
        repo_url = api.create_repo(
            token=token,
            name=repo_name or model_name,
            organization=organization,
            exist_ok=True,
        )
        repo = huggingface_hub.Repository(
            str(tmpdir), clone_from=repo_url, use_auth_token=token
        )

        tmp_path = Path(tmpdir)
        zip_run(run, tmp_path)
        create_info_file(tmp_path)

        # this method automatically puts large files (>10MB) into git lfs
        repo.push_to_hub(commit_message=commit or "Automatic push from classy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="The model you want to upload")
    parser.add_argument(
        "--organization",
        help="[optional] the name of the organization where you want to upload the model",
    )
    parser.add_argument(
        "--name",
        help="Optional name to use when uploading to the HuggingFace repository",
    )
    parser.add_argument(
        "--commit", help="Commit message to use when pushing to the HuggingFace Hub"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    upload(args.model_name, args.organization, args.name, args.commit)


if __name__ == "__main__":
    main()
