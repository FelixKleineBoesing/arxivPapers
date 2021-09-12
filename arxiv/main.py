from pathlib import Path
import pandas as pd
from ogb.nodeproppred import Evaluator

from arxiv.extraction import download_arxiv_dataset, get_masks
from arxiv.modelling import train_model, build_model


def main():
    dataset = download_arxiv_dataset(Path("..",  "data", "raw"))
    graph = dataset[0]

    number_nodes = dataset.n_nodes
    number_node_features = dataset.n_node_features
    number_classes = dataset.n_labels

    idx = dataset.get_idx_split()
    masks = get_masks(idx, number_nodes)
    evaluator = Evaluator("ogbn-arxiv")

    model, optimizer, loss = build_model(number_nodes=number_nodes, number_features=number_node_features, num_classes=number_classes)
    loss, acc = train_model(model, optimizer, loss, graph, evaluator=evaluator, masks=masks)


if __name__ == "__main__":
    main()