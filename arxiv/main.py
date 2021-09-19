from pathlib import Path
import pandas as pd
from ogb.nodeproppred import Evaluator

from arxiv.evaluate import evaluate_model
from arxiv.extraction import download_arxiv_dataset, get_masks
from arxiv.modelling import train_model, build_model

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


def main():
    dataset = download_arxiv_dataset(Path("..",  "data", "raw"))
    graph = dataset[0]

    number_nodes = dataset.n_nodes
    number_node_features = dataset.n_node_features
    number_classes = dataset.n_labels

    idx = dataset.dataset.get_idx_split()
    masks = get_masks(idx, number_nodes)
    evaluator = Evaluator("ogbn-arxiv")

    model = build_model(number_nodes=number_nodes, number_features=number_node_features, num_classes=number_classes)
    model = train_model(model, dataset, masks=masks)
    evl_results = evaluate_model(model, dataset, masks)
    print(evl_results)


if __name__ == "__main__":
    main()