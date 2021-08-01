from ogb.nodeproppred import NodePropPredDataset
from pathlib import Path


def download_arxiv_dataset(download_dir: Path = Path("..", "data",  "raw")):
    dataset = NodePropPredDataset(name="ogbn-arxiv", root=download_dir)
    return dataset


def get_train_test_val_idx(dataset: NodePropPredDataset):
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    return  train_idx, val_idx,  test_idx


if __name__ == "__main__":
    dataset = download_arxiv_dataset(Path("..", "data",  "raw"))