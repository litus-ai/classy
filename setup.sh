#!/bin/bash

set -e

# setup conda
source ~/miniconda3/etc/profile.d/conda.sh

## create conda env
read -rp "Enter environment name: " env_name
read -rp "Enter python version (e.g. >=3.8): " python_version
conda create -yn "$env_name" python="$python_version"
conda activate "$env_name"

# install torch
read -rp "Enter torch version (recommended 1.9.0): " torch_version
echo "Not sure which CUDA version you have? Check out https://stackoverflow.com/a/68499241/1908499"
read -rp "Enter cuda version (10.2, 11.3 or none to avoid installing cuda support): " cuda_version
if [ "$cuda_version" == "none" ]; then
    conda install -y pytorch=$torch_version torchvision cpuonly -c pytorch
else
    conda install -y pytorch=$torch_version torchvision cudatoolkit=$cuda_version -c pytorch
fi

# install python requirements
pip install -e .
classy --install-autocomplete

echo "Classy successfully installed. Don't forget to activate your environment!"
echo "$> conda activate ${env_name}"

# install contributor dependencies
read -p "Install contributor dependencies? [y/N] "
if [[ $REPLY =~ ^[Yy]$ ]]
then
  # install black
  pip install black
  # install and hook pre-commit
  pip install pre-commit
  pre-commit install
  pre-commit run --all-files
fi
