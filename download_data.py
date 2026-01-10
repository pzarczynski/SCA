import argparse
import os

import requests
from tqdm import tqdm

URL = "https://huggingface.co/datasets/zarczynski/ASCAD/resolve/main/"


def fetch_url(url, out: str, block_size=1024**2):
    r = requests.get(url, stream=True)
    r.raise_for_status()

    total_size = int(r.headers.get("content-length", 0))
    n_blocks = (total_size + block_size - 1) // block_size

    with open(out, "wb") as f:
        for data in tqdm(r.iter_content(block_size), total=n_blocks,
                         unit="MB", ascii=True):
            f.write(data)


def download_dataset(name: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fetch_url((URL + name), path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="ASCAD.h5")
    parser.add_argument("-o", "--out-dir", type=str, default=".")
    args = parser.parse_args()

    download_dataset(args.dataset, args.out_dir)
