from dgl import to_networkx
from networkx import kamada_kawai_layout
from ogb.nodeproppred import DglNodePropPredDataset
from pathlib import Path
import pandas as pd
import numpy as np
import dgl


def download_arxiv_dataset(download_dir: Path = Path("../..", "data", "raw")):
    dataset = DglNodePropPredDataset(name="ogbn-arxiv", root=download_dir)
    return dataset


def read_mappings(download_dir: Path = Path("../..", "data", "raw")):
    category_path_csv = Path(download_dir, Path("ogbn_arxiv", "mapping", "labelidx2arxivcategeory.csv.gz"))
    paper_id_path_csv = Path(download_dir, Path("ogbn_arxiv", "mapping", "nodeidx2paperid.csv.gz"))
    categories = pd.read_csv(category_path_csv)
    paper_ids = pd.read_csv(paper_id_path_csv)
    categories.columns =["ID", "category"]
    return categories, paper_ids


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


def read_meta_data(path: str = Path("../..", "data", "raw", "titleabs.tsv")):
    data = pd.read_table(path, header=None)
    data.columns = ["ID", "Title", "Abstract"]
    data.drop([0, data.shape[0]-1], axis=0, inplace=True)
    return data


def get_mapping(dataset, meta_data: pd.DataFrame, node_ids: pd.DataFrame, categories):
    meta_data = meta_data.copy()
    meta_data["ID"] = meta_data["ID"].astype(np.int64)
    meta_data.columns = ["mag_id", "title", "abstract"]
    node_ids.columns = ["ID", "mag_id"]
    categories.columns = ["label_id", "category"]
    labels = dataset.labels.numpy()
    node_ids["label_id"] = labels
    data = pd.merge(node_ids, meta_data, how="left", on="mag_id")
    data = pd.merge(data, categories, how="left", on="label_id")
    return data


def get_edges_from_dataset(dataset):
    edges = np.array([e.numpy() for e in dataset.graph[0].edges()]).transpose()
    return edges


def convert_edges_to_sigma_format(dataset):
    edges = get_edges_from_dataset(dataset)
    return [
        {"key": i, "source": row[0], "target": row[1], "attributes": {}} for i, row in enumerate(edges)
    ]


def convert_nodes_to_sigma_format(node_data: pd.DataFrame, dataset):
    node_data = node_data.to_dict("records")
    return [
        {"key": row["ID"], "attributes": {"label": row["title"], "category": ["category"]}} for row in node_data
    ]


