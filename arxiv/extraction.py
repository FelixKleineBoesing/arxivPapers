from ogb.nodeproppred import NodePropPredDataset
from pathlib import Path


def download_arxiv_dataset(download_dir: Path =Path("data",  "raw")):
    dataset = NodePropPredDataset(name="ogbg-arxiv", root="data/")




if __name__ == "__main__":
    download_arxiv_dataset()