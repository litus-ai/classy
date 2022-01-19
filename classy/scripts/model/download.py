import argparse
import json
import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import requests

from classy.utils.experiment import Experiment
from classy.utils.file import (
    CLASSY_DATE_FORMAT,
    CLASSY_HF_INFO_URL,
    CLASSY_HF_MODEL_URL,
    CLASSY_MODELS_CACHE_PATH,
    ensure_dir,
    get_md5,
)

logger = logging.getLogger(__name__)


def assert_file_exists(path: Path, md5=None):
    assert path.exists(), f"Could not find file {path}"
    if md5:
        file_md5 = get_md5(path)
        assert file_md5 == md5, "md5 for %s is %s, expected %s" % (path, file_md5, md5)


def unzip(zip_path: Path, target: Path):
    import zipfile

    """
    Unzip the contents of `zip_path` into `target`.
    """
    logger.debug(f"Unzipping {zip_path} to {target}")
    with zipfile.ZipFile(str(zip_path)) as f:
        f.extractall(target)


def download_resource(resource_url: str, output_path: Path) -> int:
    """
    Download a resource from a specific url into an output_path
    """
    import requests
    from tqdm.auto import tqdm

    req = requests.get(resource_url, stream=True)

    with output_path.open("wb") as f:
        file_size = int(req.headers.get("content-length"))
        default_chunk_size = 131072
        desc = "Downloading " + resource_url
        with tqdm(
            total=file_size, unit="B", unit_scale=True, desc=desc
        ) as progress_bar:
            for chunk in req.iter_content(chunk_size=default_chunk_size):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    progress_bar.update(len(chunk))

    req.raise_for_status()
    return req.status_code


def request_file(url, path):
    """
    A complete wrapper over download_file() that also make sure the directory of
    `path` exists, and that a file matching the md5 value does not exist.
    """
    download_resource(url, path)
    assert_file_exists(path)


def download(model_name: str, force_download: bool = False):

    if "@" in model_name:
        user_name, model_name = model_name.split("@")
    else:
        user_name = "sunglasses-ai"

    model_qualifier = f"{user_name}@{model_name}"

    # download model information
    model_info_url = CLASSY_HF_INFO_URL.format(
        user_name=user_name, model_name=model_name
    )

    # no need to remove any file under the tmp directory as it is automatically removed upon exiting this block
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        tmp_info_path = tmp_path / "info.json"
        logger.info("Attempting to download remote model information")

        try:
            request_file(model_info_url, tmp_info_path)
        except requests.exceptions.HTTPError:
            logger.error(
                f"{model_name} cannot be found in the Hugging Face model hub classy section"
            )
            return

        # load model information
        with tmp_info_path.open() as f:
            model_info_dict = json.load(f)

        # we only perform sanity / existence checks if the user has not explicitly request to re-download
        if not force_download:

            # check if a model with the same name is in the cache
            # we treat it as an Experiment whose creation date is the actual download date
            # the creation date is stored in the info file, we compare the two and decide whether the
            # downloaded model is still valid or if it has to be re-downloaded
            exp = Experiment.from_name(
                model_qualifier, exp_dir=CLASSY_MODELS_CACHE_PATH
            )

            if exp is not None:
                run = exp.last_run
                downloaded_date = run.date
                upload_date = datetime.strptime(
                    model_info_dict["upload_date"], CLASSY_DATE_FORMAT
                )

                if downloaded_date <= upload_date:
                    # model needs to be re-downloaded as a new version is on the hub
                    to_download = True
                    logger.info(
                        "Found an older version of the model in the cache, re-downloading..."
                    )
                else:
                    # check files' correctness
                    logger.info(
                        "Found a model in the cache with the same name, checking their equivalence"
                    )

                    with (run.directory / "info.json").open() as f:
                        cached_model_info_dict = json.load(f)

                    if model_info_dict["md5"] == cached_model_info_dict["md5"]:
                        logger.info(
                            "The models have the same md5, thus they should be equal. Returning..."
                        )
                        return
                    else:
                        to_download = True
                        logger.info(
                            "Found an older version of the model in cache. Downloading the new one"
                        )
            else:
                to_download = True

            if not to_download:
                return
        else:
            logger.info(
                "Skipping existence / sanity checks as --force-download was provided"
            )

        # model directory is cache_dir/model_name/date/time, to follow the experiments' convention
        now = datetime.now().strftime(CLASSY_DATE_FORMAT.replace(" ", "/"))
        model_cache_path = ensure_dir(CLASSY_MODELS_CACHE_PATH / model_qualifier / now)

        # downloading the actual model
        model_url = CLASSY_HF_MODEL_URL.format(
            user_name=user_name, model_name=model_name
        )
        tmp_model_path = tmp_path / "model.zip"
        request_file(model_url, tmp_model_path)

        # checking if the model md5 is the same declared in its info box
        downloaded_model_md5 = get_md5(tmp_model_path)
        if downloaded_model_md5 != model_info_dict["md5"]:
            logger.error(
                "The downloaded model has an md5 that is not equal to the one declared in its info box. "
                "Removing the model and returning..."
            )
            return

        # create model dir if empty and transfer the model info in the final repository
        ensure_dir(model_cache_path)
        # cannot use .rename() here as it might break with cross-link devices
        # see https://stackoverflow.com/questions/42392600
        shutil.move(tmp_info_path, model_cache_path / "info.json")

        unzip(tmp_model_path, model_cache_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        help="The model you want to download (use user@model for a specific model)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="It will download the model even if you already have it in the "
        "cache. Usually required if you interrupted the previous download.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    download(args.model_name, args.force_download)


if __name__ == "__main__":
    main()
