#!/bin/bash

# get source directory
export NAV_SUITE_ENVS_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"


# run the formatter over the repository
# check if pre-commit is installed
if ! command -v pre-commit &>/dev/null; then
    echo "[INFO] Installing pre-commit..."
    pip install pre-commit
fi
# always execute inside the Nav Suite directory
echo "[INFO] Formatting the repository..."
cd ${NAV_SUITE_ENVS_PATH}
pre-commit run --all-files
cd - > /dev/null
