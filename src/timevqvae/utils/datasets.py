import os
import tarfile
import zipfile

import pandas as pd
import requests

from timevqvae.utils.paths import get_root_dir


def download_ucr_datasets_old(
    url="https://figshare.com/ndownloader/files/37909926",
    chunk_size=128,
    zip_fname="UCR_archive.zip",
):
    """Download original UCR archive datasets."""
    dirname = str(get_root_dir().joinpath("datasets"))
    if os.path.isdir(os.path.join(dirname, "UCRArchive_2018")):
        return None

    if not os.path.isdir(dirname) or not os.path.isfile(os.path.join(dirname, zip_fname)):
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        r = requests.get(url, stream=True)
        print("downloading the UCR archive datasets...\n")
        fname = os.path.join(dirname, zip_fname)
        with open(fname, "wb") as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)

        with tarfile.open(fname) as ff:
            ff.extractall(path=dirname)
    elif os.path.isfile(str(get_root_dir().joinpath("datasets", zip_fname))):
        fname = os.path.join(dirname, zip_fname)
        with tarfile.open(fname) as ff:
            ff.extractall(path=dirname)


def download_ucr_datasets(
    url="https://figshare.com/ndownloader/files/47494442",
    chunk_size=128,
    zip_fname="UCR_archive.zip",
):
    """Download resplit UCR archive datasets for time series generation."""
    dirname = str(get_root_dir().joinpath("datasets", "UCRArchive_2018_resplit"))
    fname = os.path.join(dirname, zip_fname)

    if os.path.isdir(dirname) and len(os.listdir(dirname)) > 1:
        return None

    if not os.path.isdir(dirname) or not os.path.isfile(fname):
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        r = requests.get(url, stream=True)
        print("downloading the UCR archive datasets...\n")

        with open(fname, "wb") as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)

        with zipfile.ZipFile(fname, "r") as zz:
            zz.extractall(path=dirname)
    elif os.path.isfile(fname):
        with zipfile.ZipFile(fname, "r") as zz:
            zz.extractall(path=dirname)


def get_target_ucr_dataset_names(args):
    if args.dataset_names:
        return args.dataset_names

    data_summary_ucr = pd.read_csv(get_root_dir().joinpath("datasets", "DataSummary_UCR.csv"))
    return data_summary_ucr["Name"].tolist()
