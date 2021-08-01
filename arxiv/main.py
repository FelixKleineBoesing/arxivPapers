from pathlib import Path
import pandas as pd

from arxiv.extraction import download_arxiv_dataset, get_train_test_val_idx
from arxiv.modelling import train_model, build_model


def main():
    dataset = download_arxiv_dataset(Path("..",  "data", "raw"))
    train_idx, val_idx, test_idx = get_train_test_val_idx(dataset)
    data, label = dataset[0]
    edges = pd.DataFrame(data["edge_index"].transpose(), columns=["source", "target"])
    model, generator = build_model(nodes=data["node_feat"], edges=edges, num_classes=dataset.num_classes)
    loss, acc = train_model(model, generator, train_label=label[train_idx], train_idx=train_idx,
                            test_label=label[test_idx], test_idx=test_idx)

if __name__ == "__main__":
    main()