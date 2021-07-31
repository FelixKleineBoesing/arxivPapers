from ogb.nodeproppred import NodePropPredDataset
from pathlib import Path


def download_arxiv_dataset(download_dir: Path = Path("..", "data",  "raw")):
    _ = NodePropPredDataset(name="ogbn-arxiv", root=download_dir)


if __name__ == "__main__":
    download_arxiv_dataset(Path("..", "data",  "raw"))