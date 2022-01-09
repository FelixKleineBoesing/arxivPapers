import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
import numpy as np
import pandas as pd
import sklearn
import tqdm

from arxiv.helpers import SparseRowIndexer


def evaluate(graph, model, masks, evaluator):
    x, adj, y = graph.x, graph.a, graph.y
    p = model([x, adj], training=False)
    p = p.numpy().argmax(-1)[:, None]
    tr_mask, va_mask, te_mask = masks["train"], masks["val"], masks["test"]
    tr_auc = evaluator.eval({"y_true": y[tr_mask], "y_pred": p[tr_mask]})["acc"]
    va_auc = evaluator.eval({"y_true": y[va_mask], "y_pred": p[va_mask]})["acc"]
    te_auc = evaluator.eval({"y_true": y[te_mask], "y_pred": p[te_mask]})["acc"]
    return tr_auc, va_auc, te_auc


class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type='mean')
        self.h_feats = h_feats

    def forward(self, mfgs, x):
        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        return h


def train_model(model, graph, train_idx, val_idx = None, epochs: int = 20,
                early_stopping_patience: int = 3, batch_size: int = 1024):
    optimizer = torch.optim.Adam(model.parameters())
    accuracy_not_improved_since = 0
    train_dataloader = get_data_loader(graph, train_idx, batch_size=batch_size)
    if val_idx is not None:
        val_dataloader = get_data_loader(graph, val_idx,  batch_size=batch_size)
    else:
        val_dataloader = None

    best_accuracy = 0
    best_model_path = 'model.pt'
    for epoch in range(1, 1 + epochs):
        model.train()
        with tqdm.tqdm(train_dataloader) as tq:
            for batch, (inp_nodes, out_nodes, mfgs) in enumerate(tq):
                loss, accuracy = _train_step(model, optimizer,  mfgs)
                tq.set_postfix({'loss': '%.03f' % loss.item(), 'acc': '%.03f' % accuracy}, refresh=False)

        if val_dataloader is not None:
            val_accuracy = _eval_model(model, val_dataloader)
            print('Epoch {} Validation Accuracy {}'.format(epoch, val_accuracy))
            accuracy_not_improved_since += 1
            if best_accuracy < val_accuracy:
                accuracy_not_improved_since = 0
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), best_model_path)
            if accuracy_not_improved_since >= early_stopping_patience:
                print("Breaked")
                break
            print(f"Best Accuracy: {best_accuracy}")
            print(f"Not improved since: {accuracy_not_improved_since}")
    print(f"Best Accuracy: {best_accuracy}")


def _eval_model(model, val_dataloader=None):
    model.eval()
    val_predictions = []
    val_labels = []
    if val_dataloader is not None:
        with tqdm.tqdm(val_dataloader) as tq, torch.no_grad():
            for input_nodes, output_nodes, mfgs in tq:
                inputs = mfgs[0].srcdata['feat']
                val_labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
                val_predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
            val_predictions = np.concatenate(val_predictions)
            val_labels = np.concatenate(val_labels)
            accuracy = sklearn.metrics.accuracy_score(val_labels, val_predictions)

    return accuracy


def _train_step(model, optimizer,  mfgs):
    inputs = mfgs[0].srcdata['feat']
    labels = mfgs[-1].dstdata['label']

    predictions = model(mfgs, inputs)

    loss = F.cross_entropy(predictions, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    accuracy = sklearn.metrics.accuracy_score(labels.cpu().numpy(),
                                              predictions.argmax(1).detach().cpu().numpy())
    return loss, accuracy


def _get_shuffled_batches(x, adj, y, batch_size):
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    batches = int(len(x) / batch_size) + 1
    row_indexer = SparseRowIndexer(adj)

    for batch in range(batches):
        min_index = batch * batch_size
        max_index = batch * batch_size + batch_size
        sliced_adj = row_indexer[np.arange(min_index, max_index)]
        yield batch, ((x[min_index:max_index, :], sliced_adj.toarray()), y[min_index:max_index, :])


def get_data_loader(graph, ids, batch_size: int = 1024):
    sampler = dgl.dataloading.MultiLayerNeighborSampler([4, 4])
    train_dataloader = dgl.dataloading.NodeDataLoader(
        graph, ids, sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0
    )
    return train_dataloader
