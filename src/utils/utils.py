""" utils modules for making PaDME easier on Jupyter Notebooks """

from pathlib import Path
import zipfile
import os
import yaml

from IPython import get_ipython


def load_notebook_params():
    with open("../datasets.yaml", "r") as f:
        notebook_params = yaml.safe_load(f)
    return notebook_params


def generate_paths(example_dataset, root_dir, notebook_params):
    filepaths = {}

    filepaths["complete_filepath"] = Path(root_dir) / Path(
        notebook_params["raw_dataset_path"]
    )

    filepaths["complete_filename"] = filepaths["complete_filepath"] / Path(
        notebook_params[example_dataset]["zipfile_name"]
    )

    filepaths["snapshots_filepath"] = (
        filepaths["complete_filepath"]
        / Path(notebook_params[example_dataset]["folder_name"])
        / Path(notebook_params[example_dataset]["folder_name"])
    )
    return filepaths


def download_dataset(filepaths, example_dataset, notebook_params, FORCE_DOWNLOAD=False):
    if not os.path.isfile(filepaths["complete_filename"]) and FORCE_DOWNLOAD:
        ipython = get_ipython()
        download_code = ipython.transform_cell(
            "! kaggle datasets download -d {notebook_params[example_dataset]['kaggle_path']}"
        )
        exec(download_code)
        os.replace(
            Path(os.getcwd()) / notebook_params[example_dataset]["zipfile_name"],
            filepaths["complete_filename"],
        )
    return


def unpack_kaggle_dataset(filepaths):
    if not os.path.exists(filepaths["complete_filepath"]):
        with zipfile.ZipFile(filepaths["complete_filename"], "r") as zip_ref:
            zip_ref.extractall(filepaths["complete_filepath"])
    return
