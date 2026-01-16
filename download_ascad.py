import argparse
import os

import urllib3
from tqdm import tqdm

DATASET_URL = "https://huggingface.co/datasets/zarczynski/ASCAD/resolve/main/"


def fetch_url(url: str, out: str, block_size:int = 1024**2) -> None:
    resp = urllib3.request("GET", url, preload_content=False)
    
    total_size = int(resp.headers.get("content-length", 0))
    n_blocks = (total_size + block_size - 1) // block_size

    with open(out, "wb") as f:
        for data in tqdm(resp.stream(block_size), total=n_blocks, unit="MB", ascii=True):
            f.write(data)

    resp.release_conn()


def download_dataset(ds: str, out: str) -> None:    
    out_dir = os.path.dirname(out)
    os.makedirs(out_dir, exist_ok=True)
    fetch_url(DATASET_URL + ds, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="v1/variable/desync0.h5")
    parser.add_argument("-o", "--out", type=str, default="data/raw/v1_var_desync0.h5")
    args = parser.parse_args()

    download_dataset(args.dataset, args.out)
