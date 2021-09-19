from ogb.nodeproppred import NodePropPredDataset
from pathlib import Path
import numpy as np
from spektral.datasets.ogb import OGB
from spektral.layers import GCNConv
from spektral.transforms import LayerPreprocess


def download_arxiv_dataset(download_dir: Path = Path("..", "data",  "raw")):
    dataset = NodePropPredDataset(name="ogbn-arxiv", root=download_dir)
    ogb_dataset = OGB(dataset, transforms=[LayerPreprocess(GCNConv)])
    return ogb_dataset


def get_masks(idx, number_nodes):
    idx_tr, idx_va, idx_te = idx["train"], idx["valid"], idx["test"]
    mask_tr = np.zeros(number_nodes, dtype=bool)
    mask_va = np.zeros(number_nodes, dtype=bool)
    mask_te = np.zeros(number_nodes, dtype=bool)
    mask_tr[idx_tr] = True
    mask_va[idx_va] = True
    mask_te[idx_te] = True
    masks = {"train": mask_tr, "val": mask_va, "test": mask_te}
    return masks
