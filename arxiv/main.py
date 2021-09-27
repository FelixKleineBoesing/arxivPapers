from pathlib import Path
import pandas as pd
import torch
from ogb.nodeproppred import Evaluator


from arxiv.extraction import download_arxiv_dataset, get_masks, get_graph_and_node_labels
from arxiv.modelling import train_model, Model



def main():
    dataset = download_arxiv_dataset(Path("..",  "data", "raw"))
    graph, node_labels = get_graph_and_node_labels(dataset)
    node_features = graph.ndata['feat']
    num_features = node_features.shape[1]
    num_classes = (node_labels.max() + 1).item()

    evaluator = Evaluator("ogbn-arxiv")
    masks = get_masks(dataset)
    model = Model(num_features, 256, num_classes)
    model = train_model(model, graph=graph, train_idx=masks["train"], val_idx=masks["val"])


if __name__ == "__main__":
    main()