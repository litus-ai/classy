import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import os

import requests
from datasets import tqdm

import logging

logger = logging.getLogger(__name__)


CLASSY_HF_MODEL_URL = "https://huggingface.co/edobobo/{model_name}/resolve/main/{model_name}.zip"
CLASSY_HF_INFO_URL = "https://huggingface.co/edobobo/{model_name}/raw/main/{model_name}.info.json"
CLASSY_MODELS_CACHE_PATH = "{home_dir}/.cache/sunglasses_ai/classy/{model_name}"


def ensure_dir(path):
    """
    Create dir in case it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_md5(path):
    """
    Get the MD5 value of a path.
    """
    with open(path, "rb") as fin:
        data = fin.read()
    return hashlib.md5(data).hexdigest()


def assert_file_exists(path, md5=None):
    assert os.path.exists(path), "Could not find file at %s" % path
    if md5:
        file_md5 = get_md5(path)
        assert file_md5 == md5, "md5 for %s is %s, expected %s" % (path, file_md5, md5)


def unzip(path, filename):
    """
    Fully unzip a file `filename` that's in a directory `dir`.
    """
    logger.debug(f"Unzip: {path}/{filename}...")
    with zipfile.ZipFile(os.path.join(path, filename)) as f:
        f.extractall(path)


def download_resource(resource_url: str, output_path: str) -> int:
    """
    Download a resource from a specific url into an output_path
    """
    req = requests.get(resource_url, stream=True)
    with open(output_path, "wb") as f:
        file_size = int(req.headers.get("content-length"))
        default_chunk_size = 131072
        desc = "Downloading " + resource_url
        with tqdm(total=file_size, unit="B", unit_scale=True, desc=desc) as progress_bar:
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


def download(model_name: str, force_download: bool):

    # download model information
    model_info_url = CLASSY_HF_INFO_URL.format(model_name=model_name)
    tmp_info_path = f"/tmp/{model_name}.info.json"
    logger.info("Attempting to download remote model information")

    try:
        request_file(model_info_url, tmp_info_path)
    except requests.exceptions.HTTPError:
        logger.error(f"{model_name} cannot be found in the Hugging Face model hub classy section")
        return

    # load model information
    with open(tmp_info_path) as f:
        model_info_dict = json.load(f)

    # check if a model with the same name is in the cache
    model_cache_path = CLASSY_MODELS_CACHE_PATH.format(home_dir=os.getenv("HOME"), model_name=model_name)
    if os.path.exists(model_cache_path) and not force_download:
        logger.info("Found a model in the cache with the same name, checking their equivalence")
        with open(model_cache_path + f"/{model_name}.info.json") as f:
            cached_model_info_dict = json.load(f)
        if model_info_dict["md5"] == cached_model_info_dict["md5"]:
            logger.info("The models have the same md5, thus they should be equal. Returning...")
            os.remove(tmp_info_path)
            return
        else:
            logger.info("Found an older version of the model in cache. Downloading the new one")

    # create model dir if empty and transfer the model info in the final repository
    ensure_dir(model_cache_path)
    os.system(f"mv {tmp_info_path} {model_cache_path}/.")

    # downloading the actual model
    model_url = CLASSY_HF_MODEL_URL.format(model_name=model_name)
    tmp_model_path = f"/tmp/{model_name}.zip"
    request_file(model_url, tmp_model_path)

    # checking if the model md5 is the same declared in its info box
    downloaded_model_md5 = get_md5(tmp_model_path)
    if downloaded_model_md5 != model_info_dict["md5"]:
        logger.error(
            "The downloaded model has an md5 that is not equal to the one declared in its info box. "
            "Removing the model and returning..."
        )
        os.remove(tmp_model_path)
        return

    # moving the final model in the correct repository and unzipping it
    model_path = model_cache_path + f"/{model_name}.zip"
    if os.path.exists(model_path):
        os.remove(model_path)  # if any
    os.system(f"mv {tmp_model_path} {model_cache_path}/.")
    unzip(model_cache_path, f"{model_name}.zip")

    # finally delete the zip file
    os.remove(model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="The model you want to download")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="It will download the model even if u already have the model in the cache",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    download(args.model_name, args.force_download)


if __name__ == "__main__":
    main()
