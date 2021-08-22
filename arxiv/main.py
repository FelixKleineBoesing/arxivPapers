from pathlib import Path
import pandas as pd

from arxiv.extraction import download_arxiv_dataset, get_masks
from arxiv.modelling import train_model, build_model


def main():
    dataset = download_arxiv_dataset(Path("..",  "data", "raw"))
    graph = dataset[0]
    x, adj, y = graph.x, graph.a, graph.y
    number_nodes = dataset.n_nodes
    number_node_features = dataset.n_node_features
    number_classes = dataset.n_labels

    idx = dataset.get_idx_split()
    masks = get_masks(idx, number_nodes)


    model, generator = build_model()
    loss, acc = train_model(model, generator, train_label=label[train_idx], train_idx=train_idx,
                            test_label=label[test_idx], test_idx=test_idx)


if __name__ == "__main__":
    main()