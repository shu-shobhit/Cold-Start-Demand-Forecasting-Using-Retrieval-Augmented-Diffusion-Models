"""Utility script for downloading benchmark datasets used by the project.

The repository only uses a subset of these datasets for RATD forecasting,
but the script is inherited from the upstream CSDI-style tooling and remains
useful for reproducing related experiments.
"""

import os
import pickle
import sys
import tarfile
import zipfile

import pandas as pd
import requests
import wget


os.makedirs("data/", exist_ok=True)
if sys.argv[1] == "physio":
    # Download and unpack the PhysioNet challenge data into the local data
    # directory. The extraction layout is preserved from the original code.
    url = "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download"
    wget.download(url, out="data")
    with tarfile.open("data/set-a.tar.gz", "r:gz") as t:
        t.extractall(path="data/physio")

elif sys.argv[1] == "pm25":
    # The PM2.5 benchmark is distributed as a zip file, so the script downloads
    # the archive bytes manually and then extracts them.
    url = "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/STMVL-Release.zip"
    urlData = requests.get(url).content
    filename = "data/STMVL-Release.zip"
    with open(filename, mode="wb") as f:
        f.write(urlData)
    with zipfile.ZipFile(filename) as z:
        z.extractall("data/pm25")

    def create_normalizer_pm25():
        """Create and persist training-set mean/std statistics for PM2.5.

        Returns:
            None: The function writes a pickle file with normalization stats.
        """

        # The original preprocessing excludes the held-out test months before
        # computing normalization statistics.
        df = pd.read_csv(
            "./data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
            index_col="datetime",
            parse_dates=True,
        )
        test_month = [3, 6, 9, 12]
        for i in test_month:
            df = df[df.index.month != i]
        mean = df.describe().loc["mean"].values
        std = df.describe().loc["std"].values
        path = "./data/pm25/pm25_meanstd.pk"
        with open(path, "wb") as f:
            pickle.dump([mean, std], f)

    create_normalizer_pm25()
