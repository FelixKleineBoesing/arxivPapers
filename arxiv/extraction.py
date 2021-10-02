from ogb.nodeproppred import DglNodePropPredDataset
from pathlib import Path
import pandas as pd
import dgl


def download_arxiv_dataset(download_dir: Path = Path("..", "data",  "raw")):
    dataset = DglNodePropPredDataset(name="ogbn-arxiv", root=download_dir)
    return dataset


def get_graph_and_node_labels(dataset: DglNodePropPredDataset):
    graph, node_labels = dataset[0]
    # Add reverse edges since ogbn-arxiv is unidirectional.
    graph = dgl.add_reverse_edges(graph)
    graph.ndata['label'] = node_labels[:, 0]
    return graph, node_labels


def get_masks(dataset):
    idx_split = dataset.get_idx_split()
    train_ids = idx_split['train']
    valid_ids = idx_split['valid']
    test_ids = idx_split['test']
    masks = {"train": train_ids, "val": valid_ids, "test": test_ids}
    return masks


def read_meta_data(path: str = Path("..", "data", "raw", "titleabs.tsv")):
    data = pd.read_table(path, header=None)
    data.columns = ["ID", "Title", "Abstract"]
    #data.drop(0, axis=0, inplace=True)



if __name__ == "__main__":
    read_meta_data()
